"""
main.py — FastAPI backend for the Civic Services Q&A Agent.

Wraps query_engine.answer_question() as a POST /ask endpoint, serves a
minimal chat UI at /, and (optionally) logs each query for later analysis.

Run locally:
    export GEMINI_API_KEY=your_key_here
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""

import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query_engine import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("civic-qa")

app = FastAPI(title="Civic Services Q&A Agent")

# Loosen CORS for the hackathon demo; tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.time()
    try:
        result = answer_question(request.question)
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        if "429" in str(e):
            logger.warning("rate limited: query=%r", request.question)
            return AskResponse(
                answer="We're getting a lot of questions right now and hit a temporary limit. Please wait about 30 seconds and try again.",
                sources=[],
                latency_ms=latency_ms,
            )
        logger.exception("unexpected error: query=%r", request.question)
        raise
    latency_ms = int((time.time() - start) * 1000)

    logger.info(
        "query=%r sources=%r latency_ms=%d",
        request.question,
        result["sources"],
        latency_ms,
    )
    # Optional: this is where a BigQuery insert would go, logging
    # (question, sources, latency_ms, timestamp) for the "smarter
    # communities" analytics angle. Skip it if you're short on time —
    # the /ask endpoint works fully without it.

    return AskResponse(answer=result["answer"], sources=result["sources"], latency_ms=latency_ms)


CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Indore Civic Services Q&A</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; background: #f7f7f8; }
  h1 { font-size: 1.4rem; }
  #chat { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  .msg { padding: 12px 16px; border-radius: 10px; line-height: 1.4; white-space: pre-wrap; }
  .user { background: #dbeafe; align-self: flex-end; }
  .bot { background: #ffffff; border: 1px solid #e5e7eb; }
  .sources { font-size: 0.8rem; color: #6b7280; margin-top: 6px; }
  form { display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid #d1d5db; }
  button { padding: 10px 18px; border-radius: 8px; border: none; background: #2563eb; color: white; cursor: pointer; }
  button:disabled { background: #93c5fd; }
</style>
</head>
<body>
<h1>Indore Civic Services Assistant</h1>
<p>Ask about water connections, property tax, birth/death certificates, or civic complaints.</p>
<div id="chat"></div>
<form id="form">
  <input type="text" id="question" placeholder="e.g. How do I get a new water connection?" autocomplete="off" required />
  <button type="submit" id="submitBtn">Ask</button>
</form>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('question');
const submitBtn = document.getElementById('submitBtn');

function addMessage(text, cls, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  if (sources && sources.length) {
    const src = document.createElement('div');
    src.className = 'sources';
    src.textContent = 'Sources: ' + sources.join(', ');
    div.appendChild(src);
  }
  chat.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, 'user');
  input.value = '';
  submitBtn.disabled = true;

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    addMessage(data.answer, 'bot', data.sources);
  } catch (err) {
    addMessage('Something went wrong. Please try again.', 'bot');
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return CHAT_HTML
