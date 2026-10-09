"""
RAG pipeline for the AI Laboratory Assistant Chatbot.

Runs entirely on local models served by Ollama (no external API, no API key):
  - nomic-embed-text for embeddings
  - phi3:mini for answer generation

Responsibilities:
  - split lab documents (project requirements, procedures, equipment guides,
    troubleshooting notes) into retrievable chunks
  - embed and store those chunks in a local Chroma collection
  - retrieve the most relevant chunks for a student question
  - generate a grounded answer using only the retrieved content, with a
    clear "not confident" path when nothing relevant is found
"""

import os
import uuid
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "phi3:mini"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # characters of overlap between chunks
TOP_K = 4                 # number of chunks retrieved per question
CONFIDENCE_FLOOR = 0.32   # below this similarity, treat as "no confident match"

DB_PATH = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "lab_knowledge_base"

ollama_client = ollama.Client(host=OLLAMA_HOST)


class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function backed by a local Ollama model."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [
            ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)["embedding"]
            for text in input
        ]


chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH),
    settings=chromadb.Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=OllamaEmbeddingFunction(),
)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def extract_text(file_path: Path) -> str:
    """Extract raw text from a .pdf or .txt/.md lab document."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries where possible."""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_document(file_path: Path, source_label: str | None = None) -> int:
    """
    Chunk a document and add it to the knowledge base.
    Returns the number of chunks added.
    """
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    label = source_label or file_path.name
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": label, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def ingest_directory(directory: Path) -> dict[str, int]:
    """Ingest every .pdf/.txt/.md file in a directory. Returns {filename: chunk_count}."""
    results = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in (".pdf", ".txt", ".md"):
            results[path.name] = ingest_document(path)
    return results


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a laboratory assistant chatbot for a college course.
Answer the student's question using ONLY the reference material provided below.
Reference material comes from the course's own project requirements, procedures,
equipment documentation, and troubleshooting notes.

Rules:
- If the reference material answers the question, give a clear, direct answer
  and mention which source(s) it came from.
- If the reference material does NOT contain enough information to answer
  confidently, say so plainly and recommend the student check with their
  instructor or TA. Do not guess or use outside knowledge.
- Keep answers concise and practical, written for a student in the middle of
  lab work.
"""


def retrieve(question: str, top_k: int = TOP_K) -> dict:
    """Query the knowledge base and return matched chunks with distances."""
    results = collection.query(query_texts=[question], n_results=top_k)
    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def answer_question(question: str) -> dict:
    """
    Full RAG turn: retrieve relevant chunks, decide confidence, generate answer.
    Returns {"answer": str, "sources": list[str], "escalate": bool}.
    """
    if collection.count() == 0:
        return {
            "answer": "The knowledge base is empty. Ask your team to ingest the "
                      "lab's documents before using the chatbot.",
            "sources": [],
            "escalate": True,
        }

    retrieved = retrieve(question)
    documents = retrieved["documents"]
    distances = retrieved["distances"]

    # Chroma's default distance is smaller-is-more-similar (cosine distance).
    best_distance = min(distances) if distances else 1.0
    confident = bool(documents) and best_distance <= (1 - CONFIDENCE_FLOOR)

    if not confident:
        return {
            "answer": "I don't have enough information in this lab's materials to "
                      "answer that confidently. Please check with your instructor or TA.",
            "sources": [],
            "escalate": True,
        }

    sources = sorted({m["source"] for m in retrieved["metadatas"]})
    context = "\n\n---\n\n".join(documents)

    response = ollama_client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference material:\n{context}\n\nStudent question: {question}"},
        ],
        options={"temperature": 0.2},
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "escalate": False,
    }
