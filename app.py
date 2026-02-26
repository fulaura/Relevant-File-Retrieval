from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from database import database_setup
from document_load_pipe import discover_files, process_document
from main import build_llms, llm_flow

st.set_page_config(page_title="Document Retrieval Assistant", layout="wide")

EXTENSION_OPTIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".html"]
DEFAULT_EXTENSION_SET = {".pdf", ".docx", ".txt", ".md"}


@st.cache_resource(show_spinner=False)
def init_models(stream: bool = False):
    """Instantiate LLM clients once per Streamlit session."""
    return build_llms(stream=stream)


def init_state() -> None:
    if "deps" not in st.session_state:
        llm_client, silent_llm, action_llm = init_models(stream=False)
        conn, cur = database_setup.db_connect()
        if not (conn and cur):
            st.session_state["deps"] = None
            return
        st.session_state["deps"] = {
            "conn": conn,
            "cur": cur,
            "llm": llm_client,
            "silent_llm": silent_llm,
            "action_llm": action_llm,
            "top_k": 5,
            "chunks_per_doc": 2,
        }

    st.session_state.setdefault("history_records", [])
    st.session_state.setdefault("last_docs", [])
    st.session_state.setdefault("chat_log", [])
    st.session_state.setdefault("doc_paths_text", "test")
    st.session_state.setdefault("custom_extensions", "")
    st.session_state.setdefault("selected_extensions", list(DEFAULT_EXTENSION_SET))
    st.session_state.setdefault("chat_mode", "Auto")


def ingest_documents(paths: List[str], extensions: List[str]) -> None:
    deps = st.session_state["deps"]
    if not deps:
        st.error("Database connection unavailable.")
        return

    try:
        files = discover_files(paths, extensions)
    except ValueError as exc:
        st.error(str(exc))
        return

    if not files:
        st.warning("No files found for the given paths/extensions.")
        return

    progress = st.progress(0, text="Preparing to ingest documents...")
    status = st.empty()

    for idx, file_path in enumerate(files, start=1):
        try:
            process_document(Path(file_path), deps["conn"], deps["cur"])
            status.info(f"Ingested: {file_path}")
        except Exception as exc:  # noqa: BLE001
            status.error(f"Failed {file_path}: {exc}")
        progress.progress(idx / len(files), text=f"Processed {idx}/{len(files)} files")

    status.success("Ingestion completed.")
    progress.empty()


def empty_database() -> None:
    deps = st.session_state["deps"]
    if not deps:
        st.error("Database connection unavailable.")
        return
    cur = deps["cur"]
    conn = deps["conn"]
    try:
        cur.execute("TRUNCATE TABLE documents;")
        conn.commit()
        st.success("Database cleared.")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        st.error(f"Failed to empty database: {exc}")


def render_chat_interface() -> None:
    deps = st.session_state["deps"]
    if not deps:
        st.error("Database connection unavailable.")
        return

    st.subheader("Chat")
    for message in st.session_state["chat_log"]:
        role = message["role"]
        with st.chat_message(role):
            st.write(message["content"])
            if role == "assistant" and message.get("sources"):
                st.caption(message["sources"])

    prompt = st.chat_input("Ask about your documents...")
    if not prompt:
        return

    deps["top_k"] = st.session_state["chat_top_k"]
    deps["chunks_per_doc"] = st.session_state["chat_chunks"]

    mode = st.session_state.get("chat_mode", "Auto")
    force_action = None
    if mode == "Direct only":
        force_action = "direct"
    elif mode == "Retrieve list":
        force_action = "rdi"
    elif mode == "Answer with docs":
        force_action = "rcfd"

    user_entry = {"role": "user", "content": prompt}
    st.session_state["chat_log"].append(user_entry)
    with st.chat_message("user"):
        st.write(prompt)

    state = {
        "input": prompt,
        "input_source": "user",
        "force_action": force_action,
        "auto_mode": mode == "Auto",
    }
    with st.spinner("Thinking..."):
        final_state, history = llm_flow(
            state,
            deps,
            st.session_state["history_records"],
            st.session_state["last_docs"],
        )
        st.session_state["history_records"] = history

    response = final_state.get("context", "")
    sources = final_state.get("sources")
    assistant_entry = {
        "role": "assistant",
        "content": response,
        "sources": sources,
    }
    st.session_state["chat_log"].append(assistant_entry)
    with st.chat_message("assistant"):
        st.write(response)
        if sources:
            st.caption(sources)


def sidebar_controls() -> None:
    st.sidebar.header("Chat Settings")
    st.session_state.setdefault("chat_top_k", 5)
    st.session_state.setdefault("chat_chunks", 2)
    st.session_state["chat_top_k"] = st.sidebar.slider("Top K Documents", 1, 10, st.session_state["chat_top_k"])
    st.session_state["chat_chunks"] = st.sidebar.slider("Chunks per Document", 1, 5, st.session_state["chat_chunks"])
    mode_options = ["Auto", "Direct only", "Retrieve list", "Answer with docs"]
    current_mode = st.session_state.get("chat_mode", "Auto")
    if current_mode not in mode_options:
        current_mode = "Auto"
    st.session_state["chat_mode"] = st.sidebar.selectbox(
        "Chat Mode",
        mode_options,
        index=mode_options.index(current_mode),
    )

    if st.sidebar.button("Reset Chat", use_container_width=True):
        st.session_state["chat_log"] = []
        st.session_state["history_records"] = []
        st.session_state["last_docs"] = []
        st.sidebar.success("Chat reset.")

    st.sidebar.divider()

    st.sidebar.header("Document Loader")
    st.session_state["doc_paths_text"] = st.sidebar.text_area(
        "Document paths (one per line)",
        st.session_state["doc_paths_text"],
        height=100,
    )

    st.sidebar.caption("Toggle the file types you want to ingest.")
    selected_extensions: List[str] = []
    for ext in EXTENSION_OPTIONS:
        key = f"ext_toggle_{ext.replace('.', '')}"
        toggled = st.sidebar.toggle(ext.upper(), value=(ext in DEFAULT_EXTENSION_SET), key=key)
        if toggled:
            selected_extensions.append(ext)
    st.session_state["selected_extensions"] = selected_extensions

    st.session_state["custom_extensions"] = st.sidebar.text_input(
        "Custom extensions (comma separated)",
        st.session_state["custom_extensions"],
    )

    if st.sidebar.button("Load Documents", use_container_width=True):
        paths = [p.strip() for p in st.session_state["doc_paths_text"].splitlines() if p.strip()]
        custom_text = st.session_state["custom_extensions"].strip()
        custom_exts = [
            ext if ext.startswith('.') else f".{ext}"
            for ext in (item.strip() for item in custom_text.split(','))
            if ext
        ] if custom_text else []
        extensions = sorted(set(selected_extensions + custom_exts))
        if not paths:
            st.sidebar.warning("Add at least one document path before loading.")
        elif not extensions:
            st.sidebar.warning("Select at least one extension before loading.")
        else:
            ingest_documents(paths, extensions)

    if st.sidebar.button("Empty Database", type="primary", use_container_width=True):
        empty_database()


def main() -> None:
    init_state()
    sidebar_controls()

    if st.session_state["deps"] is None:
        st.error("Unable to establish database connection. Check server settings.")
        return

    st.title("Document Retrieval Assistant")
    st.write("Load documents into the vector store, then use the chat to ask grounded questions.")

    render_chat_interface()


if __name__ == "__main__":
    main()
