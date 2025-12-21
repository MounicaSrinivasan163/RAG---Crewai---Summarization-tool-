# crew/rag_crew.py
from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2
)

def detect_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["disadvantage", "drawback", "limitation", "negative"]):
        return "List ONLY the disadvantages or negative aspects."
    if any(w in q for w in ["advantage", "benefit", "merit"]):
        return "List ONLY the advantages or benefits."
    if any(w in q for w in ["steps", "process", "how to"]):
        return "Explain the process step by step."
    if any(w in q for w in ["difference", "compare"]):
        return "Provide a clear comparison."
    return "Answer the question directly."


def summarize_chunks_task(context):
    """
    Generate a high-quality, ChatGPT-like answer
    using retrieved RAG context.
    """

    chunks = context.get("retrieved_chunks", [])
    summary_length = context.get("summary_length", 200)
    query = context.get("query", "the user question")

    if not chunks:
        return {"summary": "I could not find relevant information in the uploaded documents."}

    # ---- Build clean context ----
    clean_context = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        if text.strip():
            clean_context.append(text.strip())

    context_text = "\n\n".join(clean_context)

    # ---- ChatGPT-style prompt ----
    intent_instruction = detect_intent(query)

    prompt = f"""
You are a knowledgeable and helpful AI assistant.

Your task is to answer the user's question using ONLY the provided context.
Do NOT mention that you are summarizing or that this comes from documents.

User question:
{query}

Guidelines:
- Focus directly on answering the question
- Be clear, structured, and easy to understand
- Use bullet points or short paragraphs where appropriate
- Avoid unnecessary definitions or background unless required
- Do not include legal, historical, or administrative details unless relevant
- If multiple viewpoints or aspects exist, organize them clearly
- Keep the response around {summary_length} words

Context:
{context_text}

Additional instruction:
{intent_instruction}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
    except Exception as e:
        summary = f"Error generating response: {e}"

    return {"summary": summary}
