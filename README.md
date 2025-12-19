# 📘RAG - DOC Search, Summarization & Evaluation Tool (RAG System)

## 🧩 Problem Definition

In many real-world scenarios, users need to quickly search through large volumes of documents (books, reports, PDFs, notes) and obtain concise, meaningful summaries of relevant content. Traditional keyword-based search fails to capture semantic meaning and does not provide synthesized answers.

This project solves the problem by building a **Retrieval-Augmented Generation (RAG)** system that:
- Semantically indexes documents
- Retrieves the most relevant chunks for a user query
- Generates a concise summary using an LLM
- Evaluates summary quality using ROUGE metrics

The system is designed to be **persistent, scalable, and production-aligned**, where indexed documents remain searchable across sessions.

---

## 🎯 Objectives

- Enable semantic search over uploaded documents
- Avoid duplicate indexing using content hashing
- Generate query-focused summaries
- Evaluate summaries using automatic and human evaluation
- Build an industry-grade RAG pipeline with clean modular design

---

## 🧪 Functional Requirements

- Upload and chunk multiple document types (PDF, TXT, CSV, DOCX)
- Generate embeddings and store them in a vector database
- Deduplicate content before indexing
- Perform semantic similarity search
- Generate summaries using an LLM
- Evaluate summaries using ROUGE scores
- Persist indexed data across sessions

---

## 🛠️ Tools & Technologies Used

| Category | Tools |
|------|------|
| Frontend | Streamlit |
| Vector Database | Pinecone |
| Embeddings | Local / Free Embedding Model |
| LLM | OpenAI-compatible LLM (via Agent abstraction) |
| Chunking | Custom text chunker |
| Evaluation | ROUGE |
| Hashing | SHA-256 |
| Environment | Python, dotenv |

---

## 🧠 System Architecture & Approach

### 1️⃣ Document Ingestion
- Documents are uploaded via Streamlit UI
- Each document is split into overlapping chunks
- Each chunk is hashed using **SHA-256** to create a stable ID

### 2️⃣ Deduplicated Indexing
- Before indexing, Pinecone is checked for existing chunk IDs
- Duplicate chunks are skipped automatically
- New chunks are embedded and upserted in batches

### 3️⃣ Semantic Retrieval
- User query is embedded
- Top-K similar chunks are retrieved from Pinecone
- Retrieval works even if indexing is not done in the current session

### 4️⃣ Summarization
- Retrieved chunks are normalized safely
- Combined context is sent to the LLM
- Summary length is controlled by user input

### 5️⃣ Evaluation
- Generated summaries are evaluated using ROUGE metrics
- Manual human evaluation guidelines are also provided

---

## ✨ Key Features

- ✅ Persistent semantic search (session-independent)
- ✅ SHA-256 based deduplication
- ✅ Batched Pinecone upserts (safe for size limits)
- ✅ Robust chunk normalization (prevents NoneType errors)
- ✅ Explicit user-triggered summarization
- ✅ Automatic + manual evaluation support
- ✅ Clean, modular, production-ready codebase
- ✅ Document-level filtering in retrieval

---

## 🧾 Project Structure

```
RAG/
│
├── app.py # Streamlit application entry point
├── reference_summaries.py # reference text for evaluation
│
├── vectorstore/
│ ├── indexer.py # Chunk embedding & Pinecone indexing
│ ├── retriever.py # Semantic retrieval logic
│ ├── embeddings.py # Embedding generation (local/free)
│ └── pinecone_client.py # Pinecone index existence checks
│
├── utils/
│ ├── file_loader.py # File parsing & chunking
│ └── hashing.py # SHA-256 content hashing
│
├── crew/
│ └── rag_crew.py # LLM / Agent abstraction
│
├── evaluation/
│ └── rouge_eval.py # ROUGE evaluation logic
│
├── .env # Environment variables
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```
## ⚙️ Environment Variables
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_index_name
```

## ▶️ How to Run the Project
```
pip install -r requirements.txt
streamlit run app.py
```


---

## 🧪 Evaluation Metrics

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Human evaluation:
  - Relevance
  - Coverage
  - Coherence
  - Correctness

---

## 🧠 Design Decisions (Why This Approach)

- **Vector DB (Pinecone)**: Enables scalable semantic search
- **Hash-based IDs**: Prevents duplicate embeddings
- **Chunking with overlap**: Preserves semantic continuity
- **Session-independent retrieval**: Aligns with real-world RAG systems
- **Modular architecture**: Easy to extend or replace components

---

## 🚀 Future Enhancements

- Multi-document comparative summaries
- Search accuracy metrics (Precision@K, Recall@K)
- Feedback-based re-ranking
- UI-based evaluation dashboards

---

## 📌 Use Cases

- Book summarization
- Research assistance
- Knowledge base search
- Academic and enterprise document analysis
- Interview-ready RAG project demonstration

---

## 👩‍💻 Author

**Mounica Srinivasan**  
| Aspiring Data Scientist  
RAG • NLP • Vector Databases • LLM Applications


