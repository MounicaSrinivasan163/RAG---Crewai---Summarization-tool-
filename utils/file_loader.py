from pypdf import PdfReader
from io import BytesIO
import streamlit as st
import pandas as pd

def load_file(uploaded_file, chunk_size=300, overlap=50):
    text = ""

    if uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(BytesIO(uploaded_file.read()))
        pages_text = [page.extract_text() for page in pdf.pages]
        text = "\n".join([p for p in pages_text if p])
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        rows = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
        text = "\n".join(rows)
    elif uploaded_file.name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return []

    words = text.split()
    chunks = []
    progress_bar = st.progress(0)
    total_chunks = max(1, len(words) // chunk_size + (1 if len(words) % chunk_size else 0))

    start, chunk_num = 0, 1
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({"id": f"{uploaded_file.name}_chunk_{chunk_num}", "text": chunk_text})
        start += chunk_size - overlap
        chunk_num += 1
        progress_bar.progress(min(chunk_num / total_chunks, 1.0))

    progress_bar.empty()
    return chunks

