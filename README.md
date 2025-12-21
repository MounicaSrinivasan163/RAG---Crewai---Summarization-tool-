# 📘 RAG-based Document Search, Summarization & Evaluation System
---
Streamlit APP link: https://rag-pinecone-doc-index-summarization-tool.streamlit.app/

---

A **production-style Retrieval-Augmented Generation (RAG) application** built with **Streamlit**, **Pinecone**, **LangChain**, and **OpenAI**, designed to ingest documents, perform **hybrid retrieval**, generate **ChatGPT-like answers**, and support both **automatic and human evaluation**.

This system supports **multi-topic content ingestion** and is not limited to a single domain.

---

## 🚀 Key Highlights

- Multi-format document ingestion (PDF, DOCX, TXT, CSV)
- Full-document deduplication using content hashing
- Persistent document registry across sessions
- Vector-based semantic retrieval (Pinecone)
- Hybrid retrieval (Vector + BM25-style keyword matching)
- ChatGPT-style answer generation (answer-focused, not generic summaries)
- Question-intent-aware responses (advantages, disadvantages, steps, comparisons)
- ROUGE-based automatic evaluation
- Industry-style human evaluation stored in CSV
- CrewAI is **NOT used** (custom LangChain orchestration instead)

---

## 🧠 Architecture Overview

User Query  
↓  
Hybrid Retrieval (Vector Similarity + Keyword Matching)  
↓  
Context Assembly  
↓  
LLM Answer Generation (ChatGPT-style)  
↓  
Evaluation (ROUGE + Human Review)

---

## 🗂️ Project Structure

```
RAG/
│
├── app.py # Streamlit application (UI + orchestration)
│
├── docs_loader.py # Persistent document registry (CSV-based)
│
├── reference_summaries.py # Gold reference answers for evaluation
│
├── vectorstore/
│ ├── embeddings.py # Embedding generation (OpenAI)
│ ├── indexer.py # Chunk upsert + deduplication (Pinecone)
│ └── retriever.py # Hybrid retrieval logic
│ └── bm25_store.py # bm25
│ └── pinecone_client.py # connect to Pinecone index vectorDB
│
├── crew/
│ └── rag_crew.py # Prompt-engineered RAG answer generation
│
├── evaluation/
│ └── rouge_eval.py # ROUGE score evaluation
│
├── utils/
│ ├── file_loader.py # Document parsing & chunking
│ └── hashing.py # Content hash for deduplication
│
├── indexed_documents.csv # Persistent indexed document registry
├── human_evaluations.csv # Stored human evaluation results
│
├── requirements.txt
└── README.md

```


---

## 📚 Features in Detail

### 1️⃣ Document Indexing
- Upload one or more documents
- Chunking with overlap for context preservation
- Batched embedding generation
- Safe Pinecone upserts (batch size controlled)
- Full-document deduplication using content hash
- Persistent document registry stored in CSV

---

### 2️⃣ Persistent Document Selection
- Previously indexed documents are available across sessions
- Dropdown shows **document names**, not hashes
- Supports:
  - Single-document querying
  - Cross-document querying (“All Documents”)

---

### 3️⃣ Hybrid Retrieval (Vector + Keyword)
- Semantic similarity via embeddings
- Keyword relevance via BM25-style matching
- Improves factual grounding and intent alignment
- Reduces irrelevant chunk retrieval

---

### 4️⃣ High-Quality Answer Generation
- ChatGPT-style responses
- Answer-focused (not summary-heavy)
- Uses only retrieved context
- Query-intent aware:
  - Advantages / Disadvantages
  - Steps / Processes
  - Comparisons
  - Direct factual answers
- Clean, structured output using bullets or short paragraphs

---

### 5️⃣ Evaluation Layer

#### 🔹 Automatic Evaluation
- ROUGE-1, ROUGE-2, ROUGE-L
- Compared against gold reference answers

#### 🔹 Human Evaluation (Industry Style)
- Ratings for:
  - Relevance
  - Coverage
  - Correctness
  - Faithfulness (hallucination check)
  - Coherence
- Evaluator notes
- Stored persistently in CSV with timestamps

---

## 🛠️ Tech Stack

- UI: Streamlit
- LLM: OpenAI (via LangChain)
- Embeddings: OpenAI
- Vector Database: Pinecone
- Retrieval: Hybrid (Vector + BM25-style)
- Evaluation: ROUGE + Human Review
- Persistence: CSV-based registry

---

## 📈 Use Cases

- Internal knowledge base Q&A
- Learning assistant across mixed topics
- RAG system prototyping
- Interview-ready GenAI project
- Evaluation framework experimentation

---

## 🔮 Future Enhancements

- Chunk-level citation highlighting
- Retrieval confidence scoring
- Cross-encoder re-ranking
- Follow-up question memory
- Adaptive top-k retrieval

---

## 👤 Author Notes

This project was built with emphasis on:
- Correct RAG principles
- Industry-aligned evaluation


It demonstrates **practical GenAI engineering**, not just API usage.
