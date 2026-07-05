#!/bin/sh
set -e

if [ ! -f civic_kb.index ]; then
  echo "No index found — building it now..."
  python ingest.py
fi

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
