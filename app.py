import streamlit as st
import fitz
import numpy as np
import faiss
import requests
from groq import Groq
import os
import re

# ================================
# SETUP
# ================================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
HF_API_KEY = st.secrets["HF_API_KEY"]

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# ================================
# EMBEDDING FUNCTION (NO TORCH)
# ================================
def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    return np.array(response.json())

# ================================
# PDF TEXT EXTRACTION
# ================================
def extract_pdf_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================================
# TEXT CHUNKING
# ================================
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

# ================================
# BUILD FAISS INDEX
# ================================
def build_faiss(chunks):
    embeddings = embed(chunks)

    mask = ~np.isnan(embeddings).any(axis=1)
    embeddings = embeddings[mask]
    chunks = [chunks[i] for i in range(len(chunks)) if mask[i]]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks

# ================================
# RETRIEVAL
# ================================
def retrieve(query, index, chunks, k=3):
    q_emb = embed([query])
    scores, idxs = index.search(q_emb, k)

    results = []
    for i, score in zip(idxs[0], scores[0]):
        results.append({"chunk": chunks[i], "score": float(score)})
    return results

# ================================
# RAG ANSWERING
# ================================
def answer_with_rag(question, index, chunks):
    retrieved = retrieve(question, index, chunks, k=3)

    context = "\n\n".join([f"Chunk {i+1}:\n{r['chunk']}" for i, r in enumerate(retrieved)])

    prompt = f"""
Use ONLY the context to answer.
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

# ================================
# STREAMLIT UI
# ================================
st.title("📘 AI Tutor – PDF RAG Assistant (Groq + FAISS + HF Embeddings)")
st.write("Upload PDF → Ask questions → AI answers using your syllabus.")

uploaded_file = st.file_uploader("📄 Upload your Syllabus PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully! Extracting text...")

    pdf_text = extract_pdf_text(uploaded_file)
    chunks = split_text(pdf_text)
    index, chunks = build_faiss(chunks)

    st.success(f"PDF processed! Total chunks created: {len(chunks)}")

    question = st.text_input("Ask a question:")

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
