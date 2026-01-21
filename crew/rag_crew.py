# crew/rag_crew.py

from langchain_openai import ChatOpenAI

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0  # 🔒 reduce hallucination
)


def detect_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["disadvantage", "drawback", "limitation", "negative"]):
        return "List the disadvantages or negative aspects."
    if any(w in q for w in ["advantage", "benefit", "merit"]):
        return "List the advantages or benefits."
    if any(w in q for w in ["steps", "process", "how to"]):
        return "Explain the steps."
    if any(w in q for w in ["difference", "compare"]):
        return "Provide a comparison."
    return "Answer the question directly."


def summarize_chunks_task(context):
    """
    Strict RAG Answering:
    - Uses ONLY retrieved document content
    - No external knowledge
    - No meta or disclaimer sentences
    """

    chunks = context.get("retrieved_chunks", [])
    summary_length = context.get("summary_length", 200)
    query = context.get("query", "")

    # 🚨 No retrieved chunks → hard refusal
    if not chunks:
        return {
            "summary": "No relevant information found in the provided documents."
        }

    # ---------------- Clean Context ----------------
    clean_context = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        if text and text.strip():
            clean_context.append(text.strip())

    if not clean_context:
        return {
            "summary": "No relevant information found in the provided documents."
        }

    context_text = "\n\n".join(clean_context)
    intent_instruction = detect_intent(query)

    # ---------------- PROMPT ----------------
    prompt = f"""
You are a document-grounded AI assistant.

CRITICAL RULES:
- Answer using ONLY the information present in the Context.
- Do NOT use external knowledge or assumptions.
- Do NOT mention the context, documents, or what is missing.
- Do NOT add disclaimers such as:
  "no other information is available",
  "explicitly mentioned",
  "not provided in the context",
  or similar phrases.
- If the answer is not present at all, respond ONLY with:
  "No relevant information found in the provided documents."

User Question:
{query}

Context:
{context_text}

Answering Instructions:
{intent_instruction}

Answer Style Rules:
- Provide ONLY the requested information
- Do NOT explain omissions
- Do NOT add concluding or meta statements
- Be concise and factual
- Use bullet points or short paragraphs if helpful
- Maximum length: {summary_length} words

Final Answer:
"""

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()

        # 🔒 Final safety net
        if not summary or "no relevant information" in summary.lower():
            return {
                "summary": "No relevant information found in the provided documents."
            }

    except Exception as e:
        summary = f"Error generating response: {e}"

    return {"summary": summary}
