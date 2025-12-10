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
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Page config
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📘",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        margin: 1.5rem 0;
    }
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .chunk-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #764ba2;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("# 📘 RAG Model")
st.markdown('<p class="subtitle">Upload PDF → Ask questions → AI answers from your syllabus!</p>', unsafe_allow_html=True)

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
    
    return [c for c in chunks if len(c) > 50]

def build_faiss(chunks):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings)
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
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📄 Upload your Syllabus PDF", type=["pdf"], label_visibility="visible")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with st.spinner('🔄 Processing your PDF...'):
        pdf_text = extract_pdf_text(uploaded_file)
        chunks = split_text(pdf_text)
        index, chunks = build_faiss(chunks)
    
    st.success(f"✅ PDF processed successfully! Total chunks created: **{len(chunks)}**")
    
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("💭 Ask a question from the PDF:", placeholder="What is the course objective?")
    with col2:
        st.write("")
        st.write("")
        ask_button = st.button("🚀 Ask AI")
    
    if ask_button:
        if question.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner('🤔 Thinking...'):
                answer, sources = answer_with_rag(question, index, chunks)
            
            st.markdown("### 🤖 AI Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📌 Retrieved Chunks")
            for i, s in enumerate(sources, 1):
                with st.expander(f"📄 Chunk {i} (Relevance: {s['score']:.2f})"):
                    st.markdown(f'<div class="chunk-box">{s["chunk"]}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Please upload a PDF to get started!")
