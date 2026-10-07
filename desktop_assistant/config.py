from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_thinking_level(value: str | None, default: str | None) -> str | None:
    if value is None:
        return default
    normalized = value.strip().lower()
    mapping = {"low": "LOW", "high": "HIGH"}
    return mapping.get(normalized, default)


@dataclass
class DesktopAssistantConfig:
    """Holds runtime configuration for the desktop assistant."""

    hotkey: str = "ctrl+alt+space"
    screenshot_dir: Path = Path(tempfile.gettempdir()) / "Relevant_doc_retriever" / "screenshots"
    tesseract_cmd: str | None = None
    deepseek_api_key: str = "sk-0dcc4b30bcce4439ac37487bc19b5bfc"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    gemini_model: str = "gemini-1.5-flash"
    gemini_thinking_level: str | None = None
    gemini_safety_threshold: str = "BLOCK_NONE"
    gemini_enable_google_search: bool = False
    gemini_response_mime_type: str = "text/plain"
    top_k: int = 5
    chunks_per_doc: int = 2
    db_dsn: str | None = None

    @classmethod
    def from_env(cls) -> "DesktopAssistantConfig":
        defaults = cls()
        screenshot_dir = Path(os.getenv("ASSISTANT_SCREENSHOT_DIR", defaults.screenshot_dir))
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", defaults.deepseek_api_key)
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", defaults.deepseek_base_url)
        deepseek_model = os.getenv("DEEPSEEK_MODEL", defaults.deepseek_model)
        gemini_model = os.getenv("GEMINI_MODEL", defaults.gemini_model)
        thinking_level = _parse_thinking_level(os.getenv("GEMINI_THINKING_LEVEL"), defaults.gemini_thinking_level)
        safety_threshold = os.getenv("GEMINI_SAFETY_THRESHOLD", defaults.gemini_safety_threshold)
        enable_google_search = _parse_bool(
            os.getenv("GEMINI_ENABLE_GOOGLE_SEARCH"),
            defaults.gemini_enable_google_search,
        )
        response_mime = os.getenv("GEMINI_RESPONSE_MIME", defaults.gemini_response_mime_type)
        hotkey = os.getenv("ASSISTANT_HOTKEY", defaults.hotkey)
        top_k = int(os.getenv("ASSISTANT_TOP_K", defaults.top_k))
        chunks_per_doc = int(os.getenv("ASSISTANT_CHUNKS", defaults.chunks_per_doc))
        db_dsn = os.getenv("DATABASE_URL")
        return cls(
            hotkey=hotkey,
            screenshot_dir=screenshot_dir,
            tesseract_cmd=tesseract_cmd,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=deepseek_base_url,
            deepseek_model=deepseek_model,
            gemini_model=gemini_model,
            gemini_thinking_level=thinking_level,
            gemini_safety_threshold=safety_threshold,
            gemini_enable_google_search=enable_google_search,
            gemini_response_mime_type=response_mime,
            top_k=top_k,
            chunks_per_doc=chunks_per_doc,
            db_dsn=db_dsn,
        )


def load_config() -> DesktopAssistantConfig:
    config = DesktopAssistantConfig.from_env()
    config.screenshot_dir.mkdir(parents=True, exist_ok=True)
    return config
