"""
ingest.py — builds the offline knowledge base for the Civic Services Q&A Agent.

Reads every markdown file in ./knowledge_base, splits each into chunks by
section header (##), embeds each chunk with Gemini's embedding model, and
saves a FAISS index plus a metadata sidecar file for retrieval at query time.

Run once, and again any time you add or edit a document:
    export GEMINI_API_KEY=your_key_here
    python ingest.py
"""

import os
import re
import json
import glob
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # 768 / 1536 / 3072 are the recommended output sizes

KB_DIR = "knowledge_base"
INDEX_PATH = "civic_kb.index"
METADATA_PATH = "civic_kb_metadata.json"

client = genai.Client(api_key=GEMINI_API_KEY)


def load_documents(kb_dir):
    docs = []
    for path in glob.glob(os.path.join(kb_dir, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_by_headers(doc_text, source):
    """Split a markdown doc on '## ' section headers. Each chunk keeps its
    section title, which doubles as retrieval context and as the citation
    shown to the citizen later."""
    sections = re.split(r"\n(?=## )", doc_text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"#{1,2}\s*(.+)", section)
        title = title_match.group(1).strip() if title_match else source
        chunks.append({"source": source, "section": title, "text": section})
    return chunks


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Embed a batch of texts with Gemini's embedding model."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return np.array([e.values for e in result.embeddings], dtype="float32")


def build_index():
    docs = load_documents(KB_DIR)
    if not docs:
        raise RuntimeError(
            f"No .md files found in {KB_DIR}/ — put your knowledge base documents there first."
        )

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_by_headers(doc["text"], doc["source"]))

    print(f"Loaded {len(docs)} documents, split into {len(all_chunks)} chunks.")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    # Normalize so inner product search behaves like cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index to {INDEX_PATH} and metadata to {METADATA_PATH}.")


if __name__ == "__main__":
    build_index()
