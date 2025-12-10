import streamlit as st
import fitz
import numpy as np
import faiss
import requests
from groq import Groq
import re
import json

# -----------------------------------------------
# 1️⃣ LOAD API KEYS
# -----------------------------------------------
GROQ_KEY = st.secrets["GROQ_API_KEY"]
HF_API_KEY = st.secrets["HF_API_KEY"]

client = Groq(api_key=GROQ_KEY)

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# -----------------------------------------------
# 2️⃣ EMBEDDINGS (ROBUST FIXED VERSION)
# -----------------------------------------------
def embed(texts):
    """Return float32 embeddings with guaranteed 2D shape."""
    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        HF_EMBED_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    try:
        raw = response.json()
    except json.JSONDecodeError:
        st.error("❌ HF returned non-JSON response.")
        st.stop()

    # Normalize output into 2D list
    cleaned = []

    # HF returns one of:
    # case A: [[vector], [vector], ...]
    # case B: [vector] when only 1 input
    # case C: [[[vector]]] wrapped
    if isinstance(raw, list):
        # If first element is NOT a list → it's a single vector
        if len(raw) > 0 and not isinstance(raw[0], list):
            cleaned.append(raw)  # convert to 2D
        else:
            # multiple embeddings
            for item in raw:
                if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
                    cleaned.append(item[0])   # unwrap [[vector]] → vector
                else:
                    cleaned.append(item)
    else:
        st.error("❌ Unexpected HF embedding format:\n" + str(raw))
        st.stop()

    arr = np.array(cleaned, dtype=np.float32)

    # Ensure shape = (#chunks, embedding_dim)
    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)

    return arr

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

    return [c for c in chunks if len(c) > 40]

# -----------------------------------------------
# 5️⃣ BUILD FAISS VECTOR INDEX
# -----------------------------------------------
def build_faiss(chunks):
    embeddings = embed(chunks)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks

# -----------------------------------------------
# 6️⃣ RETRIEVE TOP CHUNKS
# -----------------------------------------------
def retrieve(query, index, chunks, k=3):
    q_emb = embed([query]).astype("float32")
    scores, idxs = index.search(q_emb, k)

    results = []
    for i, score in zip(idxs[0], scores[0]):
        if i < len(chunks):
            results.append({"chunk": chunks[i], "score": float(score)})
    return results

# -----------------------------------------------
# 7️⃣ RAG ANSWERING
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

    if len(chunks) == 0:
        st.error("❌ PDF has no readable text.")
        st.stop()

    index, chunks = build_faiss(chunks)

    st.success(f"PDF processed! Total chunks: {len(chunks)}")
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
