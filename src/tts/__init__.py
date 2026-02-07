"""
Text-to-Speech module for ConversaVoice.

Provides Azure Neural TTS integration with SSML support
for emotional prosody control.
"""

from .ssml_builder import SSMLBuilder, ProsodyProfile, ProsodySettings, PROSODY_PROFILES
from .azure_client import AzureTTSClient, TTSError

__all__ = [
    "SSMLBuilder",
    "ProsodyProfile",
    "ProsodySettings",
    "PROSODY_PROFILES",
    "AzureTTSClient",
    "TTSError",
]
