"""
FastAPI backend for the AI Laboratory Assistant Chatbot.

Endpoints:
  POST /ingest   - upload a lab document (.pdf, .txt, .md) into the knowledge base
  POST /chat     - ask a question, get a grounded answer + sources
  GET  /health   - basic liveness check
  GET  /stats    - how many chunks are currently in the knowledge base
"""

import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

import rag  # noqa: E402  (import after load_dotenv so OLLAMA_HOST is set, if overridden)

app = FastAPI(title="AI Laboratory Assistant Chatbot")

# Allow the local static frontend (or any origin, for a class demo) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    escalate: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {"chunks_in_knowledge_base": rag.collection.count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chunk_count = rag.ingest_document(tmp_path, source_label=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    if chunk_count == 0:
        raise HTTPException(status_code=422, detail="No extractable text found in that file.")

    return {"filename": file.filename, "chunks_added": chunk_count}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag.answer_question(request.question)
    return ChatResponse(**result)
