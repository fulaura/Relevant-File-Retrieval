from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import keyboard

from desktop_assistant.config import DesktopAssistantConfig
from . import gemini_client
from .ocr_service import OCRService
from .retrieval import RetrievalEngine
from .screenshot import capture_fullscreen


class HotkeyController:
    def __init__(
        self,
        hotkey: str,
        screenshot_dir: Path,
        ocr: OCRService,
        retriever: RetrievalEngine,
        overlay_callback: Callable[[str, list[dict]], None],
        status_callback: Callable[[str], None],
        config: DesktopAssistantConfig,
    ) -> None:
        self.hotkey = hotkey
        self.dir = screenshot_dir
        self.ocr = ocr
        self.retriever = retriever
        self.overlay_callback = overlay_callback
        self.status_callback = status_callback
        self.config = config
        self._lock = threading.Lock()
        self._hotkey_ref: int | None = None

    def start(self) -> None:
        if self._hotkey_ref is not None:
            return
        self._hotkey_ref = keyboard.add_hotkey(self.hotkey, self._handle_hotkey)
        logging.info("Registered hotkey %s", self.hotkey)

    def stop(self) -> None:
        if self._hotkey_ref is None:
            return
        keyboard.remove_hotkey(self._hotkey_ref)
        self._hotkey_ref = None
        logging.info("Hotkey listener removed")

    def _handle_hotkey(self) -> None:
        if not self._lock.acquire(blocking=False):
            logging.info("Hotkey pressed while processing; ignoring.")
            return
        self.status_callback("Capturing screen...")
        threading.Thread(target=self._process_capture, daemon=True).start()

    def _process_capture(self) -> None:
        try:
            screenshot_path = self.dir / "capture.png"
            capture_fullscreen(screenshot_path)
            self.status_callback("Extracting text with OCR...")
            text = self.ocr.extract_text(screenshot_path)
            if not text.strip():
                logging.warning("OCR returned no text")
                self.status_callback("No text detected in screenshot.")
                return
            self.status_callback("Building search query...")
            query = gemini_client.build_query(text, self.config)
            self.status_callback("Searching documents...")
            docs = self.retriever.search(query)
            self.overlay_callback(text, docs)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Capture pipeline failed: %s", exc)
            self.status_callback(f"Capture failed: {exc}")
        finally:
            self._lock.release()
