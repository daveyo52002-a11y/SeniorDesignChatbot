"""Streamlit chatbot with streaming, retrieved references, feedback and session memory."""
import json
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
st.set_page_config(page_title="AI Laboratory Assistant", page_icon="🧪")
st.title("🧪 AI Laboratory Assistant")
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "auth_email" not in st.session_state:
    st.session_state.auth_email = ""
if "pending_email" not in st.session_state:
    st.session_state.pending_email = ""

if not st.session_state.auth_token:
    st.subheader("Wayne State email login")
    st.info("Enter your @wayne.edu email to receive a six-digit login code.")
    with st.form("email_form"):
        email = st.text_input("Wayne State email", value=st.session_state.pending_email)
        requested = st.form_submit_button("Send verification code")
    if requested:
        try:
            response = requests.post(f"{API_BASE}/auth/request-code",
                                     json={"email": email.strip()}, timeout=25)
            response.raise_for_status()
            st.session_state.pending_email = email.strip().lower()
            st.success("Check your Wayne State inbox for the six-digit code.")
        except requests.RequestException as exc:
            st.error(f"Could not send code: {exc}")
    if st.session_state.pending_email:
        with st.form("verify_form"):
            code = st.text_input("Six-digit code", max_chars=6)
            verified = st.form_submit_button("Verify and log in")
        if verified:
            try:
                response = requests.post(f"{API_BASE}/auth/verify",
                                         json={"email": st.session_state.pending_email, "code": code},
                                         timeout=15)
                response.raise_for_status()
                st.session_state.auth_token = response.json()["access_token"]
                st.session_state.auth_email = st.session_state.pending_email
                st.session_state.pending_email = ""
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Verification failed: {exc}")
    st.stop()

AUTH_HEADERS = {"Authorization": f"Bearer {st.session_state.auth_token}"}
with st.sidebar:
    st.caption(f"Signed in as {st.session_state.auth_email}")
    if st.button("Log out"):
        try:
            requests.post(f"{API_BASE}/auth/logout", headers=AUTH_HEADERS, timeout=5)
        except requests.RequestException:
            pass
        for key in ("auth_token", "auth_email", "pending_email", "messages"):
            st.session_state.pop(key, None)
        st.rerun()
st.caption("Answers grounded in uploaded laboratory documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"])
    if uploaded and st.button("Ingest document"):
        try:
            with st.spinner("Indexing document..."):
                res = requests.post(f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())}, headers=AUTH_HEADERS, timeout=180)
                res.raise_for_status()
                st.success(f"Indexed {res.json()['chunks_added']} chunks from {uploaded.name}")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    try:
        res = requests.get(f"{API_BASE}/stats", headers=AUTH_HEADERS, timeout=5)
        res.raise_for_status()
        st.metric("Indexed chunks", res.json()["chunks_in_knowledge_base"])
    except requests.RequestException:
        st.warning("Backend unavailable")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

def show_references(refs):
    if refs:
        with st.expander("Retrieved source excerpts (check against original files)"):
            for ref in refs:
                st.markdown(f"**[{ref['id']}] {ref['source']} — page {ref['page']}**")
                st.caption(ref['excerpt'])

def feedback_controls(msg):
    if msg.get("role") != "assistant" or not msg.get("question") or msg.get("escalate"):
        return
    mid = msg["id"]
    left, right = st.columns([1, 1])
    for col, label, rating in ((left, "👍 Helpful", 1), (right, "👎 Not helpful", -1)):
        if col.button(label, key=f"vote_{mid}_{rating}"):
            try:
                res = requests.post(f"{API_BASE}/feedback", json={
                    "message_id": mid, "question": msg["question"],
                    "answer": msg["content"], "rating": rating}, headers=AUTH_HEADERS, timeout=10)
                res.raise_for_status()
                st.toast("Feedback saved")
            except requests.RequestException:
                st.error("Could not save feedback")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("escalate"):
            st.warning(msg.get("content", msg.get("answer", "")))
        else:
            st.markdown(msg.get("content", msg.get("answer", "")))
        if msg["role"] == "assistant":
            show_references(msg.get("references", []))
            feedback_controls(msg)

if question := st.chat_input("Ask about your course or lab..."):
    # Capture a small preceding history BEFORE appending the new question.
    history = [{"role": m["role"], "content": m.get("content", "")}
               for m in st.session_state.messages if m.get("role") in ("user", "assistant")][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        refs = []
        escalate = False
        finished = False
        try:
            with requests.post(f"{API_BASE}/chat/stream", json={"question": question, "history": history},
                               headers=AUTH_HEADERS, stream=True, timeout=(10, 180)) as res:
                res.raise_for_status()
                for line in res.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("type")
                    if kind == "references":
                        refs = event.get("references", [])
                    elif kind == "token":
                        answer += event.get("text", "")
                        placeholder.markdown(answer + "▌")
                    elif kind == "done":
                        answer = event.get("answer", answer)
                        refs = event.get("references", refs)
                        escalate = event.get("escalate", False)
                        finished = True
                    elif kind == "error":
                        raise RuntimeError(event.get("message", "Generation failed"))
            if not finished:
                raise RuntimeError("Incomplete response from backend")
            if escalate:
                placeholder.warning(answer)
            else:
                placeholder.markdown(answer)
            show_references(refs)
            msg = {"role": "assistant", "id": str(uuid.uuid4()), "content": answer,
                   "question": question, "references": refs, "escalate": escalate}
            st.session_state.messages.append(msg)
            feedback_controls(msg)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            placeholder.error(f"Chat request failed: {exc}")
