import streamlit as st
import re
import faiss
import numpy as np
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pptx import Presentation
import docx
from groq import Groq
from fpdf import FPDF

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI Study Assistant (RAG)", layout="wide")
st.title("📚 AI Study Assistant (RAG)")
st.write("Upload your study materials and ask questions about them.")

# ---------------------------
# Load GROQ API Key from Secrets
# ---------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

groq_api_key = st.secrets["GROQ_API_KEY"]

# ---------------------------
# Education Level Setting
# ---------------------------
education_level = st.selectbox(
    "Select Education Level",
    [
        "Primary School",
        "Middle School",
        "Secondary School",
        "College/High School",
        "Undergraduate",
        "Graduate"
    ],
    index=3
)

# ---------------------------
# File Upload
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload Files (PDF, PPTX, DOCX, XLSX, TXT, MD)",
    accept_multiple_files=True,
    type=["pdf", "pptx", "docx", "xlsx", "txt", "md"]
)

# ---------------------------
# Helper Functions
# ---------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text(file):
    name = file.name
    ext = name.split(".")[-1].lower()
    raw = file.read()
    file.seek(0)

    if ext == "pdf":
        reader = PdfReader(BytesIO(raw))
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    if ext == "pptx":
        prs = Presentation(BytesIO(raw))
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texts.append(shape.text)
        return "\n".join(texts)

    if ext == "docx":
        d = docx.Document(BytesIO(raw))
        return "\n".join([p.text for p in d.paragraphs])

    if ext == "xlsx":
        sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
        text = ""
        for name, df in sheets.items():
            text += f"\nSheet: {name}\n"
            text += df.to_csv(index=False)
        return text

    if ext in ["txt", "md"]:
        return raw.decode("utf-8", errors="ignore")

    return ""

def chunk_text(text, chunk_size=3000, overlap=400):
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def build_faiss_index(chunks):
    model = load_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

def retrieve(query, index, chunks, k=6):
    model = load_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

def call_groq(system_prompt, user_prompt):
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# ---------------------------
# Build Knowledge Base
# ---------------------------

if st.button("🔧 Build Knowledge Base"):
    if not uploaded_files:
        st.warning("Upload files first.")
    else:
        all_chunks = []
        for file in uploaded_files:
            text = extract_text(file)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            st.error("No readable content found.")
        else:
            index = build_faiss_index(all_chunks)
            st.session_state["index"] = index
            st.session_state["chunks"] = all_chunks
            st.success("Knowledge Base Ready ✅")

# ---------------------------
# Ask Question
# ---------------------------

question = st.text_input("Ask a Question About Your Study Material")

if st.button("💬 Ask Question"):
    if "index" not in st.session_state:
        st.warning("Build Knowledge Base first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        retrieved_chunks = retrieve(
            question,
            st.session_state["index"],
            st.session_state["chunks"]
        )

        context = "\n\n---\n\n".join(retrieved_chunks)

        system_prompt = f"""
You are an AI Study Assistant.

IMPORTANT RULES:
- Only answer using the provided context.
- If answer is not found, say clearly.
- Adapt explanation to: {education_level}.
- Provide a short summary at the end.
"""

        user_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}
"""

        with st.spinner("Generating answer..."):
            answer = call_groq(system_prompt, user_prompt)

        st.subheader("📘 Answer")
        st.write(answer)

        st.download_button(
            "Download as Markdown",
            answer,
            file_name="summary.md"
        )

        pdf_bytes = generate_pdf(answer)
        st.download_button(
            "Download as PDF",
            pdf_bytes,
            file_name="summary.pdf",
            mime="application/pdf"
        )
