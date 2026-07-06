Indore Civic Services Q&A Agent

A RAG-powered assistant that answers citizen questions about Indore Municipal Corporation (IMC) services - water connections, property tax, birth/death certificates, and civic complaints/grievances - in plain language, with source citations, in both English and Hindi.

Built for the Gen AI Academy APAC Hackathon (Hack2Skill), Problem Statement 2: AI for Better Living and Smarter Communities.

---
The problem

Citizens navigating IMC services face scattered, inconsistent information across portals, PDFs, and helplines. Simple questions like "how do I get a new water connection" require digging through multiple sources. This agent answers directly, in the citizen's own language, citing exactly which official process it's drawing from - and honestly says "I don't know, contact the helpline" rather than guessing when it lacks the information.

---
Architecture

Offline ingestion pipeline: civic service documents → Cloud Storage → chunked and embedded (Gemini embedding model) → FAISS vector index.

Online query pipeline: citizen question → FastAPI (Cloud Run) → retrieval + cross-encoder reranking against the vector index → Gemini generates a grounded, cited answer → returned to the citizen.

Key design choices:

Grounded answers only - Gemini is constrained to answer solely from retrieved civic documents, and explicitly say when it doesn't have enough information, rather than guessing on matters like tax deadlines or legal processes.
Bilingual by default - automatically detects and responds in the citizen's language (English or Hindi), including Hinglish input, since that's how many citizens naturally type.
Accurate source attribution - sources shown are only the documents Gemini actually used to answer, not just everything retrieved.

---
Tech stack

Backend: FastAPI, Python
AI: Gemini API (embeddings + generation)
Retrieval: FAISS (vector search) + cross-encoder reranking (sentence-transformers)
Deployment: Docker, Google Cloud Run
Storage: Google Cloud Storage (source documents)

---
Knowledge base

Currently covers 5 core IMC services: new water connections, property tax, birth certificates, death certificates, and civic complaints/grievance escalation (Indore 311 app + MP CM Helpline 181). Designed to be easily extended — drop a new markdown document into knowledge_base/ and re-run ingest.py.

---
Running locally

bashpip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python ingest.py          # builds the FAISS index (run once)
uvicorn main:app --reload --port 8000

Then open http://localhost:8000.

---
Roadmap

BigQuery logging of queries to surface which civic services generate the most confusion - a decision-intelligence signal for the city, not just a chatbot.
Expanded knowledge base covering additional IMC services.