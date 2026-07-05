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
- If the context does not contain enough information to answer, say so plainly and suggest the IMC helpline instead of guessing.
- Always mention which service document(s) your answer is based on.
- Write in plain, simple language, avoid bureaucratic jargon.
- Keep the answer focused and actionable: what to do, in what order.
"""


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
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


def answer_question(query):
    candidates = retrieve(query)
    if not candidates:
        return {
            "answer": "I couldn't find anything relevant in the civic knowledge base. Please contact the IMC helpline: 1800-233-5522.",
            "sources": [],
        }
    top_chunks = rerank(query, candidates)
    answer = generate_answer(query, top_chunks)
    sources = sorted({c["source"] for c in top_chunks})
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    while True:
        q = input("\nAsk a civic question (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = answer_question(q)
        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])
