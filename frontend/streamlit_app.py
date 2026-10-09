"""
Streamlit frontend for the AI Laboratory Assistant Chatbot.

Talks to the FastAPI backend's /chat and /ingest endpoints. Run the backend
first (see backend/README or the main README), then:

    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# Change this if your backend runs somewhere other than localhost:8000
# (e.g. a Jetstream2 instance's IP).
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Laboratory Assistant Chatbot", page_icon="🧪")

st.title("🧪 AI Laboratory Assistant Chatbot")
st.caption("Ask about project requirements, procedures, equipment, or troubleshooting for this lab.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can answer questions about this lab's project requirements, "
                "procedures, equipment, and troubleshooting. What do you need help with?"
            ),
            "escalate": False,
            "sources": [],
        }
    ]

# --- Sidebar: upload documents into the knowledge base -----------------------
with st.sidebar:
    st.header("Knowledge base")
    st.caption("Upload this lab's documents (.pdf, .txt, .md).")
    uploaded = st.file_uploader(
        "Add a document", type=["pdf", "txt", "md"], accept_multiple_files=False
    )
    if uploaded is not None:
        if st.button("Ingest this file"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    res = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
                    if res.ok:
                        data = res.json()
                        st.success(f"Added {data['chunks_added']} chunks from {data['filename']}.")
                    else:
                        st.error(res.json().get("detail", "Upload failed."))
                except requests.exceptions.RequestException:
                    st.error("Could not reach the backend. Is it running?")

    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
        st.metric("Chunks in knowledge base", stats["chunks_in_knowledge_base"])
    except requests.exceptions.RequestException:
        st.warning("Backend not reachable.")

# --- Chat history --------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("escalate"):
            st.warning(msg["content"])
        else:
            st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Source: " + ", ".join(msg["sources"]))

# --- Chat input ------------------------------------------------------------------
if question := st.chat_input("Ask a question about the lab..."):
    st.session_state.messages.append({"role": "user", "content": question, "escalate": False, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_BASE}/chat", json={"question": question}, timeout=30)
                if res.ok:
                    data = res.json()
                    if data["escalate"]:
                        st.warning(data["answer"])
                    else:
                        st.write(data["answer"])
                    if data["sources"]:
                        st.caption("Source: " + ", ".join(data["sources"]))
                    st.session_state.messages.append({"role": "assistant", **data})
                else:
                    detail = res.json().get("detail", "Something went wrong.")
                    st.error(detail)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": detail, "escalate": True, "sources": []}
                    )
            except requests.exceptions.RequestException:
                msg = "Could not reach the chatbot backend. Is it running?"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "escalate": True, "sources": []}
                )
