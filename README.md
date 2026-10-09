# AI Laboratory Assistant Chatbot

A RAG-based chatbot that answers student questions about one lab course's
project requirements, procedures, equipment, and troubleshooting/data
interpretation guidance, grounded in that course's own documents. Runs
entirely on local models via Ollama (Phi-3-mini for answers, nomic-embed-text
for retrieval), so there's no API key and no per-query cost.

## How it works

1. Lab documents (`.pdf`, `.txt`, `.md`) are chunked and embedded locally,
   then stored in a Chroma vector database.
2. When a student asks a question, the backend retrieves the most relevant
   chunks and asks the local LLM to answer using only that retrieved content.
3. If nothing relevant is found, the bot says so and points the student to
   their instructor/TA instead of guessing.

```
lab-chatbot/
  backend/
    main.py                 FastAPI app (endpoints: /ingest, /chat, /health, /stats)
    rag.py                  chunking, embedding, retrieval, answer generation (Ollama)
    load_knowledge_base.py  CLI script to bulk-load backend/knowledge_base/
    knowledge_base/         drop the lab's source documents here
    requirements.txt
    .env.example
  frontend/
    index.html              standalone HTML/JS chat UI (no build step)
    streamlit_app.py        Streamlit chat UI (Python, talks to the same API)
    requirements.txt
```

Pick one frontend, you don't need both. `index.html` needs nothing installed
beyond a browser; `streamlit_app.py` keeps everything in Python.

## Setup

### 1. Install Ollama and pull the models

On the machine or VM that will run the backend (Red Hat/RHEL, Ubuntu, or your
own machine):

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
ollama pull nomic-embed-text
```

Ollama runs as a background service on `localhost:11434` once installed; you
don't need to start it manually.

On Red Hat/RHEL, if `curl | sh` isn't allowed on your system, Ollama also
publishes a standalone Linux binary on its GitHub releases page as an
alternative.

### 2. Install Python dependencies (Python 3.10+)

```
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No `.env` edits are required unless Ollama is running somewhere other than
`localhost:11434`, in that case copy `.env.example` to `.env` and set
`OLLAMA_HOST`.

### 3. Add the lab's documents

Drop the lab's project requirement sheets, procedure guides, equipment
manuals, and troubleshooting notes (`.pdf`, `.txt`, or `.md`) into
`backend/knowledge_base/`, then load them:

```
python load_knowledge_base.py
```

Re-run this any time you add more documents.

### 4. Start the backend

```
uvicorn main:app --reload --port 8000
```

### 5. Start a frontend

HTML version: just open `frontend/index.html` in a browser.

Streamlit version:
```
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Both talk to `http://localhost:8000` by default. If the backend is hosted
elsewhere (e.g. a Jetstream2 instance), update the `API_BASE` constant near
the top of whichever frontend you're using.

## Next steps for the team

- [ ] Pick the specific lab course and gather its real documents
- [ ] Load those documents and sanity-check retrieval quality with real
      student-style questions, Phi-3-mini is small, so test it's actually
      answering well before relying on it for the demo
- [ ] Tune `CONFIDENCE_FLOOR` in `rag.py` if the bot is too willing (or too
      unwilling) to answer
- [ ] Write the 8-10 sample demo questions for the live presentation
- [ ] Confirm which frontend (HTML or Streamlit) you're standardizing on
- [ ] Set up the Jetstream2 instance (Red Hat/RHEL) and confirm Ollama runs
      on it with enough RAM for phi3:mini

## Notes on scope

This matches the proposal: one lab course, project requirements + procedures
+ equipment + troubleshooting/data-interpretation guidance, no grading logic,
no live equipment integration.
