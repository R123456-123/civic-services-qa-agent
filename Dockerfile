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
COPY ingest.py query_engine.py main.py ./

# Build the FAISS index into the image at build time (documents are static
# for the hackathon demo, so there's no need to run ingest.py at startup)
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
RUN python ingest.py

# Cloud Run injects $PORT; default to 8080 for local testing
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
