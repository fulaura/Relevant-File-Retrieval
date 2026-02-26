from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from database import database_search, database_setup
from models import encoder


ACTION_PROMPT = PromptTemplate.from_template(
    """
    Conversation history:
    {history}

    Last retrieved documents: {last_docs}

    User input:
    "{input}"

    Available actions:
    - retrieve document info from database: rdi
    - reference content from document from database: rcfd
    - stop: stop

    IMPORTANT: Use `rdi` or `rcfd` ONLY if the user explicitly mentions document, search, files, database, or references a previous document.
    If the user wants an answer without doc search, return `stop`.

    Respond ONLY with one of: rdi, rcfd, or stop.
"""
)
EXTRACT_PROMPT = PromptTemplate.from_template(
    """
    Conversation history:
    {history}

    Last retrieved documents: {last_docs}

    Extract key information from the input for semantic search embedding.
    Output a short query phrase (5-12 words) without extra text.
    Input: {input}
    Output:
"""
)
SUMMARY_PROMPT = PromptTemplate.from_template(
    """
    You are given content from several documents.
    Answer the user's question using only the document content.
    If the answer isn't in the documents, say you couldn't find it.

    User question:
    {question}

    Documents:
    {docs}
"""
)
DIRECT_PROMPT = PromptTemplate.from_template("""{input}""")

SEARCH_KEYWORDS = {
    "document",
    "documents",
    "doc",
    "file",
    "files",
    "pdf",
    "dataset",
    "search",
    "find",
    "lookup",
    "retrieve",
    "database",
    "db",
    "corpus",
    "info",
    "information",
    "details",
    "chunk",
    "section",
    "content",
}


def _filter_docs_by_reference(user_input: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return []
    normalized = user_input.lower()
    matched: List[Dict] = []
    for doc in docs:
        name = (doc.get("file_name") or "").lower()
        stem = Path(name).stem.lower() if name else ""
        candidates = [name, stem]
        if any(token and token in normalized for token in candidates):
            matched.append(doc)
    return matched


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def build_llms(stream: bool) -> Tuple[OllamaLLM, OllamaLLM, OllamaLLM]:
    callbacks = [StreamingStdOutCallbackHandler()] if stream else []
    streamed_llm = OllamaLLM(model="dolphin-llama3:8b", callbacks=callbacks)
    silent_llm = OllamaLLM(model="dolphin-llama3:8b", callbacks=[])
    action_llm = OllamaLLM(model="dolphin-llama3:8b", temperature=0.0, max_tokens=1)
    return streamed_llm, silent_llm, action_llm


def decide_action(user_input: str, action_llm: OllamaLLM, history: List[Dict], last_docs: List[Dict]) -> str:
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
    doc_labels = ", ".join(doc.get("file_name", "Unknown") for doc in last_docs) or "none"
    action = (ACTION_PROMPT | action_llm).invoke({
        "input": user_input,
        "history": history_text or "none",
        "last_docs": doc_labels,
    }).strip().lower()
    if action not in ("rdi", "rcfd", "stop"):
        action = "stop"
    return action


def extract_query(user_input: str, llm_client: OllamaLLM, history: List[Dict], last_docs: List[Dict]) -> str:
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
    doc_labels = ", ".join(doc.get("file_name", "Unknown") for doc in last_docs) or "none"
    return (EXTRACT_PROMPT | llm_client).invoke({
        "input": user_input,
        "history": history_text or "none",
        "last_docs": doc_labels,
    }).strip()


def retrieve_docs(query: str, deps: dict) -> List[Dict]:
    embedded = encoder.encode_text_transformer(query)
    raw = database_search.similar_n(deps["conn"], deps["cur"], embedded, n=deps["top_k"] * deps["chunks_per_doc"], similarity_threshold=None)
    grouped = {}
    for similarity, file_name, content, metadata, path in raw:
        doc = grouped.setdefault(path, {
            "file_name": file_name,
            "path": path,
            "metadata": metadata or {},
            "chunks": []
        })
        doc["chunks"].append({"similarity": similarity, "content": content})

    def best_score(doc):
        return min(chunk["similarity"] for chunk in doc["chunks"]) if doc["chunks"] else float("inf")

    ordered = sorted(grouped.values(), key=best_score)
    for doc in ordered:
        doc["chunks"] = doc["chunks"][: deps["chunks_per_doc"]]
        doc["content"] = "\n\n".join(chunk["content"] for chunk in doc["chunks"])
    return ordered[: deps["top_k"]]


def format_sources(docs: List[Dict]) -> str:
    if not docs:
        return "Sources: none"
    parts = []
    for doc in docs[:5]:
        similarity = doc["chunks"][0]["similarity"] if doc["chunks"] else None
        label = doc.get("file_name", "Unknown")
        if similarity is not None:
            parts.append(f"{label} (score={similarity:.4f})")
        else:
            parts.append(label)
    return "Sources: " + ", ".join(parts)


def answer_with_docs(user_input: str, docs: List[Dict], llm_client: OllamaLLM) -> str:
    paired_docs = "\n\n".join(
        f"Document: {doc.get('file_name', 'Unknown')}\nMetadata: {doc.get('metadata', {})}\nContent:\n{doc.get('content', '')}"
        for doc in docs
    )
    return (SUMMARY_PROMPT | llm_client).invoke({"question": user_input, "docs": paired_docs}).strip()


def answer_direct(user_input: str, llm_client: OllamaLLM) -> str:
    return (DIRECT_PROMPT | llm_client).invoke({"input": user_input}).strip()


def llm_flow(state: Dict, deps: dict, history: List[Dict] | None = None, last_docs: List[Dict] | None = None) -> Tuple[Dict, List[Dict]]:
    if history is None:
        history = []
    if last_docs is None:
        last_docs = []

    input_source = state.get("input_source", "user")
    user_input = state.get("input", "")
    history.append({
        "role": "user" if input_source == "user" else "assistant",
        "content": user_input
    })
    state["history"] = history

    force_action = state.get("force_action")
    if force_action == "direct":
        action = "direct"
    elif force_action in ("rdi", "rcfd"):
        action = force_action
    else:
        action = decide_action(user_input, deps["action_llm"], history, last_docs)

    lower_input = user_input.lower()
    referential_terms = ["info", "information", "detail", "details", "more", "chunk", "tell", "describe"]
    referential_terms_hit = bool(last_docs) and any(term in lower_input for term in referential_terms)
    referential_mention = any(keyword in lower_input for keyword in ["this", "that", "those", "it"])
    if action == "rdi" and referential_terms_hit:
        action = "rcfd"

    auto_mode = bool(state.get("auto_mode"))
    keyword_hit = any(keyword in lower_input for keyword in SEARCH_KEYWORDS)
    if auto_mode and action in ("rdi", "rcfd") and not (referential_mention or referential_terms_hit or keyword_hit):
        action = "direct"

    state["action"] = action

    if action in ("rdi", "rcfd"):
        extracted = extract_query(user_input, deps["silent_llm"], history, last_docs)
        state["extracted"] = extracted
        try:
            # Reuse last docs when user references earlier context and we still have them.
            reuse_last = bool(last_docs) and (referential_mention or referential_terms_hit)
            results = last_docs if reuse_last else retrieve_docs(extracted, deps)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Database search failed")
            state["context"] = f"Database search failed: {exc}"
            return state, history

        matched_refs = _filter_docs_by_reference(user_input, results)
        if matched_refs:
            results = matched_refs

        state["retrieved_docs"] = results
        last_docs[:] = results

        if action == "rdi":
            seen_files = {doc["file_name"] for doc in results}
            state["context"] = f"I found {len(seen_files)} documents that match your query: {', '.join(seen_files)}."
            return state, history

        response = answer_with_docs(user_input, results, deps["llm"])
        state["context"] = response
        state["sources"] = format_sources(results)
        return state, history

    response = answer_direct(user_input, deps["llm"])
    state["context"] = response
    return state, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the document database via LLM workflow.")
    parser.add_argument("--question", type=str, default="Search the database for information about NoSQL databases.")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--chunks-per-doc", type=int, default=2, dest="chunks_per_doc")
    parser.add_argument("--stream", action="store_true", help="Stream LLM output to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat session.")
    return parser.parse_args()


def chat_loop(deps: dict, initial_question: str | None = None) -> None:
    print("Type 'exit' to end the session. Press Enter on empty input to repeat the last question.")
    history: List[Dict] = []
    last_docs: List[Dict] = []
    last_question = initial_question

    while True:
        prompt = input("\nuser> ") if initial_question is None else initial_question
        initial_question = None

        if prompt is None or not prompt.strip():
            if not last_question:
                print("Please provide a question or type 'exit'.")
                continue
            prompt = last_question

        prompt = prompt.strip()
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Ending session.")
            break

        last_question = prompt
        state = {"input": prompt, "input_source": "user", "auto_mode": True}
        final_state, history = llm_flow(state, deps, history, last_docs)

        print("assistant>", final_state.get("context", ""))
        if final_state.get("sources"):
            print(final_state["sources"])


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    llm_client, silent_llm, action_llm = build_llms(stream=args.stream)

    conn, cur = database_setup.db_connect()
    if not (conn and cur):
        logging.error("Unable to connect to database. Exiting.")
        return

    try:
        deps = {
            "conn": conn,
            "cur": cur,
            "llm": llm_client,
            "silent_llm": silent_llm,
            "action_llm": action_llm,
            "top_k": args.top_k,
            "chunks_per_doc": args.chunks_per_doc,
        }

        if args.interactive:
            chat_loop(deps, initial_question=args.question)
        else:
            state = {"input": args.question, "input_source": "user", "auto_mode": True}
            final_state, _ = llm_flow(state, deps)

            print("\nFinal Output:\n", final_state.get("context", ""))
            if final_state.get("sources"):
                print(final_state["sources"])
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
