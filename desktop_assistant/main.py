from __future__ import annotations

import logging
import sys

from pathlib import Path

from PySide6 import QtCore, QtWidgets

try:
    from .config import DesktopAssistantConfig, load_config
    from .deepseek_client import send_chat_message, start_chat
    from .hotkey_service import HotkeyController
    from .ocr_service import OCRService
    from .overlay import ChatDialog, ResultsOverlay
    from .retrieval import RetrievalEngine
except ImportError:  # pragma: no cover - fallback for direct execution
    if __package__ in (None, ""):
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from desktop_assistant.config import DesktopAssistantConfig, load_config
        from desktop_assistant.deepseek_client import send_chat_message, start_chat
        from desktop_assistant.hotkey_service import HotkeyController
        from desktop_assistant.ocr_service import OCRService
        from desktop_assistant.overlay import ChatDialog, ResultsOverlay
        from desktop_assistant.retrieval import RetrievalEngine
    else:  # pragma: no cover
        raise


class DesktopAssistantRuntime(QtCore.QObject):
    display_requested = QtCore.Signal(str, list)
    status_requested = QtCore.Signal(str)

    def __init__(self, config: DesktopAssistantConfig) -> None:
        super().__init__()
        self.config = config
        self.ocr = OCRService(config.tesseract_cmd)
        self.retriever = RetrievalEngine(config.top_k, config.chunks_per_doc, config.db_dsn)
        self.overlay = ResultsOverlay()
        self.display_requested.connect(self.overlay.display_results)
        self.status_requested.connect(self.overlay.show_processing)
        self.overlay.chat_requested.connect(self.open_chat)
        self.chat_session = None
        self.chat_dialog: ChatDialog | None = None
        self.last_context_summary: dict | None = None

        self.hotkey = HotkeyController(
            hotkey=config.hotkey,
            screenshot_dir=config.screenshot_dir,
            ocr=self.ocr,
            retriever=self.retriever,
            overlay_callback=self._queue_display,
            status_callback=self._queue_status,
            config=config,
        )

    def start(self) -> None:
        self.hotkey.start()
        logging.info("Desktop assistant is running in the background.")

    def stop(self) -> None:
        self.hotkey.stop()
        self.retriever.close()

    # UI handlers -----------------------------------------------------
    def open_chat(self) -> None:
        context_info = self.last_context_summary
        context_text = (context_info or {}).get("llm_context") if context_info else None
        if self.chat_session is None:
            self.chat_session = start_chat(self.config, context=context_text)
        if self.chat_dialog is None:
            self.chat_dialog = ChatDialog(self._send_chat_message)
        self.chat_dialog.update_context(context_info)
        self.chat_dialog.show()
        self.chat_dialog.activateWindow()

    def _send_chat_message(self, message: str) -> str:
        if self.chat_session is None:
            context_text = (self.last_context_summary or {}).get("llm_context") if self.last_context_summary else None
            self.chat_session = start_chat(self.config, context=context_text)
        return send_chat_message(self.chat_session, message)

    def _queue_display(self, text: str, docs: list[dict]) -> None:
        self.last_context_summary = self._build_context_summary(text, docs)
        self.chat_session = None
        if self.chat_dialog is not None:
            self.chat_dialog.update_context(self.last_context_summary)
        self.display_requested.emit(text, docs)

    def _queue_status(self, message: str) -> None:
        self.status_requested.emit(message)

    def _build_context_summary(self, text: str, docs: list[dict]) -> dict:
        snippet = text.strip().replace("\n", " ")[:400] or "<empty>"
        doc_entries = []
        for doc in docs[:10]:
            sim = doc.get("chunks", [{}])[0].get("similarity")
            doc_entries.append({
                "name": doc.get("file_name", "Unknown"),
                "path": doc.get("path", ""),
                "similarity": sim if isinstance(sim, float) else None,
            })

        detail_lines = [f"OCR snippet: {snippet}"]
        if doc_entries:
            detail_lines.append("Documents:")
            for entry in doc_entries:
                sim = entry["similarity"]
                score = f" score={sim:.4f}" if isinstance(sim, float) else ""
                detail_lines.append(f"- {entry['name']}{score}\n  {entry['path']}")
        else:
            detail_lines.append("No matching documents were found.")

        detail_lines.append(
            "Guidance: Use the OCR snippet or listed documents only when the user asks about this capture or a specific file."
        )
        detail_lines.append("For unrelated questions, respond normally without forcing document references.")

        return {
            "count": len(docs),
            "docs": doc_entries,
            "llm_context": "\n".join(detail_lines),
        }


def run_assistant() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config()
    app = QtWidgets.QApplication(sys.argv)
    runtime = DesktopAssistantRuntime(config)
    runtime.start()
    try:
        sys.exit(app.exec())
    finally:
        runtime.stop()


if __name__ == "__main__":
    run_assistant()
