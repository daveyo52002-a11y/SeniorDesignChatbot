"""FastAPI endpoints for documents, grounded chat, streaming chat, and feedback."""
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
import rag

app = FastAPI(title="AI Laboratory Assistant Chatbot")
# Set allowed origins when exposing the backend to browsers.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

class HistoryMessage(BaseModel):
    role: str
    content: str = Field(max_length=2000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)

class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=12000)
    rating: int

FEEDBACK_DB = Path(__file__).parent / "feedback.sqlite3"

def feedback_connection():
    con = sqlite3.connect(str(FEEDBACK_DB), timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        message_id TEXT PRIMARY KEY, question TEXT NOT NULL,
        answer TEXT NOT NULL, rating INTEGER NOT NULL,
        created_at TEXT NOT NULL)""")
    return con

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return {"chunks_in_knowledge_base": rag.collection.count()}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported.")
    # Demo cap: 15 MB. A public deployment should also require upload authentication.
    content = await file.read(15 * 1024 * 1024 + 1)
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large (15 MB limit).")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        path = Path(tmp.name)
    try:
        chunks = rag.ingest_document(path, source_label=name)
    finally:
        path.unlink(missing_ok=True)
    if not chunks:
        raise HTTPException(422, "No extractable text found. Scanned PDFs need OCR.")
    return {"filename": name, "chunks_added": chunks}

@app.post("/chat")
def chat(request: ChatRequest):
    return rag.answer_question(request.question, [m.model_dump() for m in request.history])

@app.post("/chat/stream")
def stream_chat(request: ChatRequest):
    def events():
        try:
            for event in rag.generate(request.question, [m.model_dump() for m in request.history]):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception:
            # Log server-side exception without disclosing secrets to clients.
            import logging
            logging.exception("Streaming generation failed")
            yield json.dumps({"type": "error", "message": "Chat generation failed; check backend logs."}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.rating not in (-1, 1):
        raise HTTPException(400, "Rating must be -1 or 1")
    with feedback_connection() as con:
        con.execute("""INSERT INTO feedback(message_id, question, answer, rating, created_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating,
                           created_at=excluded.created_at""",
                    (request.message_id, request.question, request.answer, request.rating,
                     datetime.now(timezone.utc).isoformat()))
    return {"saved": True}
