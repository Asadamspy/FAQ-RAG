import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import re

# --------------------------------------------------------
# SETUP
# --------------------------------------------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]  # for Streamlit Cloud

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


st.title("📘 AI Tutor – PDF RAG Assistant (Powered by Groq + FAISS)")
st.write("Upload PDF → Ask questions → AI answers from your syllabus!")

# --------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------

def extract_pdf_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    chunks.append(current.strip())
    
    return [c for c in chunks if len(c) > 50]  # remove small ones

def build_faiss(chunks):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings)

    # remove invalid embeddings
    mask = ~np.isnan(embeddings).any(axis=1)
    embeddings = embeddings[mask]
    chunks = [chunks[i] for i in range(len(chunks)) if mask[i]]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks

def retrieve(query, index, chunks, k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)

    results = []
    for i, score in zip(idxs[0], scores[0]):
        results.append({"chunk": chunks[i], "score": float(score)})

    return results

def answer_with_rag(question, index, chunks):
    retrieved = retrieve(question, index, chunks, k=3)

    context = "\n\n".join([f"Chunk {i+1}:\n{r['chunk']}" for i, r in enumerate(retrieved)])

    prompt = f"""
Use ONLY the context to answer.
If answer not found, say: 'Information not available in the document.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content, retrieved

# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------

uploaded_file = st.file_uploader("📄 Upload your Syllabus PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully! 📄 Extracting text...")
    
    pdf_text = extract_pdf_text(uploaded_file)
    chunks = split_text(pdf_text)
    index, chunks = build_faiss(chunks)

    st.success(f"PDF processed! Total chunks created: {len(chunks)}")
    st.write("---")

    question = st.text_input("Ask a question from the PDF:")

    if st.button("Ask AI"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            answer, sources = answer_with_rag(question, index, chunks)

            st.subheader("🤖 AI Answer")
            st.write(answer)

            st.subheader("📌 Retrieved Chunks")
            for s in sources:
                st.code(s["chunk"])
