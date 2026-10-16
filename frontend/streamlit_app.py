
"""
Streamlit frontend for the AI Laboratory Assistant Chatbot.

Talks to the FastAPI backend's /chat and /ingest endpoints.
Run the backend first, then:

    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# FastAPI backend address.
# localhost works when the frontend and backend run
# on the same Jetstream2 instance.
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Laboratory Assistant Chatbot",
    page_icon="🧪"
)

st.title("🧪 AI Laboratory Assistant Chatbot")
st.caption(
    "Ask about project requirements, procedures, equipment, "
    "or troubleshooting for this lab."
)

# --- Initialize chat history ---------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can answer questions about this lab's "
                "project requirements, procedures, equipment, "
                "and troubleshooting. What do you need help with?"
            ),
            "escalate": False,
            "sources": [],
        }
    ]

# --- Sidebar: Knowledge base ----------------------------------------------

with st.sidebar:
    st.header("Knowledge base")
    st.caption("Upload this lab's documents (.pdf, .txt, .md).")

    uploaded = st.file_uploader(
        "Add a document",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False
    )

    if uploaded is not None:
        if st.button("Ingest this file"):
            with st.spinner("Processing..."):
                try:
                    files = {
                        "file": (uploaded.name, uploaded.getvalue())
                    }

                    res = requests.post(
                        f"{API_BASE}/ingest",
                        files=files,
                        timeout=60
                    )

                    if res.ok:
                        data = res.json()
                        st.success(
                            f"Added {data['chunks_added']} chunks "
                            f"from {data['filename']}."
                        )
                    else:
                        try:
                            detail = res.json().get(
                                "detail", "Upload failed."
                            )
                        except ValueError:
                            detail = "Upload failed."
                        st.error(detail)

                except requests.exceptions.RequestException:
                    st.error(
                        "Could not reach the backend. Is it running?"
                    )

    # Display knowledge-base statistics
    try:
        response = requests.get(
            f"{API_BASE}/stats",
            timeout=5
        )
        response.raise_for_status()
        stats = response.json()

        st.metric(
            "Chunks in knowledge base",
            stats.get("chunks_in_knowledge_base", 0)
        )

    except (requests.exceptions.RequestException, ValueError):
        st.warning("Backend not reachable.")

    # Clear chat history button
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# --- Display chat history --------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):

        # Handle messages safely even if content is missing
        content = msg.get("content", msg.get("answer", ""))

        if msg.get("escalate", False):
            st.warning(content)
        else:
            st.write(content)

        if msg.get("sources"):
            st.caption(
                "Source: " + ", ".join(msg["sources"])
            )

# --- Chat input ------------------------------------------------------------

if question := st.chat_input("Ask a question about the lab..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "escalate": False,
        "sources": []
    })

    with st.chat_message("user"):
        st.write(question)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                res = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=60
                )

                if res.ok:
                    data = res.json()

                    answer = data.get(
                        "answer",
                        "No response was returned."
                    )

                    escalate = data.get("escalate", False)
                    sources = data.get("sources") or []

                    # Display response
                    if escalate:
                        st.warning(answer)
                    else:
                        st.write(answer)

                    # Display sources
                    if sources:
                        st.caption(
                            "Source: " + ", ".join(sources)
                        )

                    # FIX: Store the answer using the content key
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "escalate": escalate,
                        "sources": sources
                    })

                else:
                    try:
                        detail = res.json().get(
                            "detail",
                            "Something went wrong."
                        )
                    except ValueError:
                        detail = "Something went wrong."

                    st.error(detail)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(detail),
                        "escalate": True,
                        "sources": []
                    })

            except requests.exceptions.RequestException:
                msg = (
                    "Could not reach the chatbot backend. "
                    "Is it running?"
                )

                st.error(msg)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg,
                    "escalate": True,
                    "sources": []
                })
