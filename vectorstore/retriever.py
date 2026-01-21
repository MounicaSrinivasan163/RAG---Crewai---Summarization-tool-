# vectorstore/retriever.py
import os
from pinecone import Pinecone
from vectorstore.embeddings import embed_texts

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def retrieve_chunks(
    query: str,
    doc_id: str = None,
    top_k: int = 10,
    bm25_store=None,
    rerank_top_k: int = 5
):
    """
    Hybrid retrieval with BGE reranker (Pinecone)
    """

    # -------- Vector search --------
    query_vector = embed_texts([query], input_type="query")[0]
    filter_ = {"doc_id": doc_id} if doc_id else None

    response = index.query(
        vector=query_vector,
        top_k=30,   # get more for reranking
        include_metadata=True,
        filter=filter_
    )

    vector_results = []
    if response and response.matches:
        vector_results = [
            {
                "id": m.id,
                "text": m.metadata.get("text", "")
            }
            for m in response.matches
            if "text" in m.metadata
        ]

    # -------- BM25 search --------
    bm25_results = []
    if bm25_store:
        bm25_hits = bm25_store.search(query, top_k=20)
        bm25_results = [
            {"id": c["id"], "text": c["text"]}
            for c in bm25_hits
        ]

    # -------- Merge & dedupe --------
    merged = {}
    for item in vector_results + bm25_results:
        merged[item["id"]] = item["text"]

    candidates = list(merged.values())

    if not candidates:
        return []

    # -------- BGE Reranker (Pinecone) --------
    rerank_response = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=candidates,
        top_n=rerank_top_k
    )
    
    reranked_chunks = [
        r.document["text"]
        for r in rerank_response.results
    ]
    
    return reranked_chunks
     reranked_chunks

