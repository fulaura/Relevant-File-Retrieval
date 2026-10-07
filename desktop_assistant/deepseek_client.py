from __future__ import annotations

import logging
import os
from typing import Dict, List

from openai import OpenAI

from desktop_assistant.config import DesktopAssistantConfig

QUERY_PROMPT = """
You are part of a desktop assistant that extracts intent from OCR text captured from the screen.
Rewrite the following noisy text into a short search query (<=20 words) that can be used to search documentation.
Return only the query without any quotes, preambles, or markdown formatting.
Text:
{ocr_text}
"""

CHAT_PROMPT = """
You are the Opal desktop assistant.
- Hold a natural, friendly conversation and answer general questions even if they are unrelated to the latest capture.
- When the user clearly references the current OCR snippet or matched documents, ground the answer in that context.
- If the documents do not contain the requested information, say so briefly before answering with general knowledge.
"""

_CLIENT: OpenAI | None = None
DEFAULT_API_KEY = "sk-0dcc4b30bcce4439ac37487bc19b5bfc"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def _ensure_client(config: DesktopAssistantConfig | None = None) -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = (
            getattr(config, "deepseek_api_key", None)
            or os.getenv("DEEPSEEK_API_KEY")
            or DEFAULT_API_KEY
        )
        base_url = (
            getattr(config, "deepseek_base_url", None)
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_BASE_URL
        )
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT


def _get_model(config: DesktopAssistantConfig | None = None) -> str:
    return (
        getattr(config, "deepseek_model", None)
        or os.getenv("DEEPSEEK_MODEL")
        or DEFAULT_MODEL
    )


def build_query(ocr_text: str, config: DesktopAssistantConfig) -> str:
    client = _ensure_client(config)
    model = _get_model(config)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": QUERY_PROMPT.format(ocr_text=ocr_text)}
            ],
            temperature=0.2,
        )
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        if not content and getattr(msg, "reasoning_content", None):
            content = msg.reasoning_content.splitlines()[-1].strip()
        return content.strip().strip('"').strip("'")
    except Exception as exc:
        logging.exception("DeepSeek build_query failed: %s", exc)
        return " ".join(ocr_text.split()[:15])


class DeepSeekChatSession:
    def __init__(self, client: OpenAI, config: DesktopAssistantConfig, context: str | None = None) -> None:
        self._client = client
        self._config = config
        self._model = _get_model(config)
        self._messages: List[Dict[str, str]] = [
            {"role": "system", "content": CHAT_PROMPT.strip()}
        ]
        if context:
            self._messages.append(
                {"role": "system", "content": f"Context information:\n{context}"}
            )

    def send_message(self, message: str) -> str:
        self._messages.append({"role": "user", "content": message})
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                temperature=0.7,
            )
            text = (response.choices[0].message.content or "").strip()
            self._messages.append({"role": "assistant", "content": text})
            return text
        except Exception as exc:
            logging.exception("DeepSeek send_message failed: %s", exc)
            return f"Error communicating with DeepSeek: {exc}"


def start_chat(config: DesktopAssistantConfig, context: str | None = None) -> DeepSeekChatSession:
    client = _ensure_client(config)
    return DeepSeekChatSession(client, config, context=context)


def send_chat_message(chat_session: DeepSeekChatSession, message: str) -> str:
    return chat_session.send_message(message)
