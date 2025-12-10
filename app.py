import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from groq import Groq
import os
import re
import json

# --------------------------------------------------------
# LOAD API KEYS
# --------------------------------------------------------
GROQ_KEY = st.secrets["GROQ_API_KEY"]
HF_KEY = st.secrets["HF_API_KEY"]

client = Groq(api_key=GROQ_KEY)

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_KEY}"}

st.title("📘 RAG Model")
st.write("Upload PDF → Ask questions → AI answers from your syllabus!")


# --------------------------------------------------------
# EMBEDDINGS (REPLACES SENTENCE TRANSFORMER)
# --------------------------------------------------------
def embed(texts):
    """Embedding via HuggingFace API — works on Streamlit Cloud."""
    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        HF_EMBED_URL,
        headers=HF_HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    try:
        raw = response.json()
    except:
        st.error("❌ HF API returned invalid response")
        st.stop()

    cleaned = []
    if isinstance(raw, list):
        # Single vector
        if raw and not isinstance(raw[0], list):
            cleaned.append(raw)
        else:
            for item in raw:
                if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
                    cleaned.append(item[0])
                else:
                    cleaned.append(item)

    arr = np.array(cleaned, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


# --------------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------------
def extract_pdf_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# --------------------------------------------------------
# SPLIT INTO CHUNKS
# --------------------------------------------------------
def split_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]


# --------------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------------
def build_faiss(chunks):
    vectors = embed(chunks)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, chunks


# --------------------------------------------------------
# RETRIEVE
# --------------------------------------------------------
def retrieve(query, index, chunks, k=3):
    qv = embed([query])
    scores, idxs = index.search(qv, k)

    out = []
    for i, sc in zip(idxs[0], scores[0]):
        out.append({"chunk": chunks[i], "score": float(sc)})
    return out


# --------------------------------------------------------
# RAG ANSWER
# --------------------------------------------------------
def answer_with_rag(question, index, chunks):
    retrieved = retrieve(question, index, chunks)

    context = "\n\n".join(
        f"Chunk {i+1}:\n{r['chunk']}"
        for i, r in enumerate(retrieved)
    )

    prompt = f"""
Use ONLY the context to answer.
If the answer is not found, say: "Information not available in the document."

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
    st.success("PDF uploaded successfully! Extracting text...")

    pdf_text = extract_pdf_text(uploaded_file)
    chunks = split_text(pdf_text)

    index, chunks = build_faiss(chunks)

    st.success(f"PDF processed! Total chunks: {len(chunks)}")
    st.write("---")

    question = st.text_input("Ask a question:")

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
