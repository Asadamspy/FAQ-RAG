import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from groq import Groq
import re

# -----------------------------------------------
# 1️⃣ LOAD API KEYS
# -----------------------------------------------
GROQ_KEY = st.secrets["GROQ_API_KEY"]
HF_API_KEY = st.secrets["HF_API_KEY"]

client = Groq(api_key=GROQ_KEY)

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# -----------------------------------------------
# 2️⃣ EMBEDDING FUNCTION (NO TORCH — HF HOSTED)
# -----------------------------------------------
def embed(texts):
    """Returns a 2D float32 numpy array of embeddings."""
    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    raw = response.json()

    cleaned = []
    for item in raw:
        if isinstance(item, list):
            # If HF returns [[vector]]
            if isinstance(item[0], list):
                cleaned.append(item[0])
            else:
                cleaned.append(item)
        else:
            cleaned.append(item)

    return np.array(cleaned, dtype=np.float32)

# -----------------------------------------------
# 3️⃣ PDF TEXT EXTRACTION
# -----------------------------------------------
def extract_pdf_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -----------------------------------------------
# 4️⃣ TEXT CHUNKING
# -----------------------------------------------
def split_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 40]  # remove small chunks

# -----------------------------------------------
# 5️⃣ BUILD FAISS VECTOR INDEX
# -----------------------------------------------
def build_faiss(chunks):
    embeddings = embed(chunks).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks

# -----------------------------------------------
# 6️⃣ RETRIEVE CHUNKS
# -----------------------------------------------
def retrieve(query, index, chunks, k=3):
    q_emb = embed([query]).astype("float32")
    scores, idxs = index.search(q_emb, k)

    results = []
    for i, score in zip(idxs[0], scores[0]):
        results.append({"chunk": chunks[i], "score": float(score)})
    return results

# -----------------------------------------------
# 7️⃣ RAG ANSWERING WITH GROQ LLAMA3
# -----------------------------------------------
def answer_with_rag(question, index, chunks):
    retrieved = retrieve(question, index, chunks)

    context = "\n\n".join(
        f"Chunk {i+1}:\n{r['chunk']}" for i, r in enumerate(retrieved)
    )

    prompt = f"""
Use ONLY the context below to answer the question.
If answer is not found, say: "Information not available in the document."

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

# ----------------------------------------------------
# 8️⃣ STREAMLIT UI
# ----------------------------------------------------
st.title("📘 AI Tutor – PDF RAG Assistant (Groq + FAISS + HF Embeddings)")
st.write("Upload your PDF → Ask questions → AI answers using your syllabus content.")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded ✔ Extracting text...")

    pdf_text = extract_pdf_text(uploaded_file)
    chunks = split_text(pdf_text)
    index, chunks = build_faiss(chunks)

    st.success(f"PDF processed successfully! Total chunks: {len(chunks)}")
    st.write("---")

    question = st.text_input("Ask a question from the document:")

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            answer, retrieved = answer_with_rag(question, index, chunks)

            st.subheader("🤖 AI Answer")
            st.write(answer)

            st.subheader("📌 Retrieved Chunks")
            for r in retrieved:
                st.code(r["chunk"])
