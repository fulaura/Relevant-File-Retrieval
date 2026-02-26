from __future__ import annotations

import os
from typing import List

from google import genai
from google.genai import types

from desktop_assistant.config import DesktopAssistantConfig

QUERY_PROMPT = """
You are part of a desktop assistant that extracts intent from OCR text captured from the screen.
Rewrite the following noisy text into a short search query (<=20 words) that can be used to search documentation.
Return only the query.
Text:
{ocr_text}
"""

CHAT_PROMPT = """
You are the Opal desktop assistant.
- Hold a natural, friendly conversation and answer general questions even if they are unrelated to the latest capture.
- When the user clearly references the current OCR snippet or matched documents, ground the answer in that context.
- If the documents do not contain the requested information, say so briefly before answering with general knowledge.
"""

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

_CLIENT: genai.Client | None = None


def _ensure_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        api_key = "AIzaSyAbYxuxmsHLWZEbJD_waZwHMKCHod-4588"
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required for Gemini access.")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _build_generate_config(config: DesktopAssistantConfig) -> types.GenerateContentConfig:
    tools: List[types.Tool] = []
    if config.gemini_enable_google_search:
        tools.append(types.Tool(googleSearch=types.GoogleSearch()))

    thinking_config = None
    if config.gemini_thinking_level:
        thinking_config = types.ThinkingConfig(thinking_level=config.gemini_thinking_level)

    return types.GenerateContentConfig(
        thinking_config=thinking_config,
        safety_settings=[
            types.SafetySetting(category=category, threshold=config.gemini_safety_threshold)
            for category in SAFETY_CATEGORIES
        ],
        tools=tools,
        response_mime_type=config.gemini_response_mime_type,
    )


def _collect_text(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    parts: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def build_query(ocr_text: str, config: DesktopAssistantConfig) -> str:
    client = _ensure_client()
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=QUERY_PROMPT.format(ocr_text=ocr_text))],
        )
    ]
    response = client.models.generate_content(
        model=config.gemini_model,
        contents=contents,
        config=_build_generate_config(config),
    )
    return _collect_text(response).strip()


class GeminiChatSession:
    def __init__(self, client: genai.Client, config: DesktopAssistantConfig, context: str | None = None) -> None:
        self._client = client
        self._config = config
        self._history: List[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=CHAT_PROMPT)])
        ]
        if context:
            self._history.append(
                types.Content(role="user", parts=[types.Part.from_text(text=context)])
            )

    def send_message(self, message: str) -> str:
        self._history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=self._history,
            config=_build_generate_config(self._config),
        )
        text = _collect_text(response).strip()
        self._history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=text)])
        )
        return text


def start_chat(config: DesktopAssistantConfig, context: str | None = None) -> GeminiChatSession:
    client = _ensure_client()
    return GeminiChatSession(client, config, context=context)


def send_chat_message(chat_session: GeminiChatSession, message: str) -> str:
    return chat_session.send_message(message)
