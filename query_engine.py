"""
query_engine.py — the online query pipeline for the Civic Services Q&A Agent.

Given a citizen's question:
1. Embed the query and retrieve the top-K candidate chunks from FAISS.
2. Rerank candidates with a cross-encoder for precision.
3. Pass the top reranked chunks to Gemini, constrained to answer only from
   the provided context and to cite its source document.

Run directly for a quick interactive test:
    export GEMINI_API_KEY=your_key_here
    python query_engine.py
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"  # check aistudio.google.com for the current free-tier flash model name
EMBED_DIM = 768

INDEX_PATH = "civic_kb.index"
METADATA_PATH = "civic_kb_metadata.json"

TOP_K_RETRIEVE = 8
TOP_K_RERANK = 3

client = genai.Client(api_key=GEMINI_API_KEY)
_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _chunks = json.load(f)

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SYSTEM_PROMPT = """You are a civic services assistant for Indore Municipal Corporation.
Answer the citizen's question using ONLY the context provided below.
Rules:
- If the context does not contain enough information to answer, say so plainly in the answer field and suggest the IMC helpline, and set sufficient_info to false.
- Write in plain, simple language, avoid bureaucratic jargon.
- Keep the answer focused and actionable: what to do, in what order.
- Detect the language the citizen's question is written in and respond in that SAME language.
  - If the question is written in Devanagari script, respond in Hindi (Devanagari).
  - If the question is in Hinglish, meaning it contains actual Hindi words spelled in Roman letters (for example "connection kaise milega" or "paani ka bill kitna hai"), respond in Hindi (Devanagari).
  - If the question is in English — even if the grammar is broken, informal, or non-native ("how can i get", "who can i get", missing articles, etc.) — respond in English. Grammatical imperfection alone is NEVER a reason to switch to Hindi. Only switch if the question actually contains Hindi vocabulary.
  - When genuinely unsure, default to English.
- The source documents are in English regardless of the question's language — translate the relevant facts faithfully into the response language, don't just say the documents are unavailable in that language.
- In sources_used, list ONLY the document filenames you actually drew facts from to build the answer. If you set sufficient_info to false and mention the helpline instead, sources_used should be an empty list, even if documents were provided to you.

Respond as JSON matching this shape:
{"answer": "...", "sources_used": ["kb_example.md"], "sufficient_info": true}
"""


def translate_query_for_retrieval(query):
    """The embedding model is multilingual, but the cross-encoder reranker
    is English-only. Translate non-English queries to English just for
    retrieval/reranking — the original query (and its language) is still
    what gets passed to Gemini for the final answer."""
    prompt = (
        "If the following text is not already in English, translate it to English. "
        "If it is already in English, return it unchanged. "
        "Return ONLY the text, no explanation.\n\n"
        f"Text: {query}"
    )
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text.strip()


def embed_query(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    vec = np.array([result.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve(query, k=TOP_K_RETRIEVE):
    query_vec = embed_query(query)
    scores, indices = _index.search(query_vec, k)
    return [_chunks[i] for i in indices[0] if i != -1]


def rerank(query, candidates, k=TOP_K_RERANK):
    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:k]]


def generate_answer(query, context_chunks):
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']} - {c['section']}]\n{c['text']}" for c in context_chunks
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nCitizen's question: {query}"

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    try:
        parsed = json.loads(response.text)
        return {
            "answer": parsed.get("answer", "").strip(),
            "sources_used": parsed.get("sources_used", []),
        }
    except (json.JSONDecodeError, AttributeError):
        # Fallback if Gemini ever returns malformed JSON — don't crash the request
        return {"answer": response.text.strip(), "sources_used": []}


def answer_question(query):
    retrieval_query = translate_query_for_retrieval(query)

    candidates = retrieve(retrieval_query)
    if not candidates:
        return {
            "answer": "I couldn't find anything relevant in the civic knowledge base. Please contact the IMC helpline: 1800-233-5522.",
            "sources": [],
        }
    top_chunks = rerank(retrieval_query, candidates)
    result = generate_answer(query, top_chunks)  # original query, so response language matches
    return {"answer": result["answer"], "sources": result["sources_used"]}


if __name__ == "__main__":
    while True:
        q = input("\nAsk a civic question (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = answer_question(q)
        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])
