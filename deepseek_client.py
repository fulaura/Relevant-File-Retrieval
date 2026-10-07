"""Re-export DeepSeek client from desktop_assistant package."""
from desktop_assistant.deepseek_client import (
    DeepSeekChatSession,
    build_query,
    send_chat_message,
    start_chat,
)

__all__ = ["DeepSeekChatSession", "build_query", "send_chat_message", "start_chat"]
