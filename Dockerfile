# Dockerfile — Civic Services Q&A Agent

FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / sentence-transformers at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the cross-encoder model at build time so the container
# doesn't hit Hugging Face on every cold start
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY knowledge_base/ ./knowledge_base/
COPY ingest.py query_engine.py main.py start.sh ./
RUN chmod +x start.sh

# Cloud Run injects $PORT; default to 8080 for local testing.
# GEMINI_API_KEY is set as a runtime environment variable in the Cloud Run
# console — the index is built on first container startup, not at build time,
# so no build-time secret is needed.
ENV PORT=8080
EXPOSE 8080

CMD ["./start.sh"]
