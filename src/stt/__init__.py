"""Speech-to-Text module with local and cloud options."""

from .whisper_client import WhisperClient, STTError
from .groq_whisper_client import GroqWhisperClient

__all__ = ["WhisperClient", "GroqWhisperClient", "STTError"]
