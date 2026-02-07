"""
Async Orchestrator for ConversaVoice.

Manages the real-time pipeline: Microphone → Whisper → LLM → TTS → Speaker
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from enum import Enum


class PipelineState(Enum):
    """States of the voice assistant pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class PipelineResult:
    """Result from a single pipeline cycle."""

    user_input: str
    assistant_response: str
    style: Optional[str] = None
    pitch: Optional[str] = None
    rate: Optional[str] = None
    is_repetition: bool = False
    latency_ms: float = 0.0


class Orchestrator:
    """
    Async orchestrator for the voice assistant pipeline.

    Coordinates: Whisper (STT) → Groq (LLM) → Azure (TTS)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            session_id: Session ID for conversation memory
            on_state_change: Callback when pipeline state changes
            on_transcription: Callback when user speech is transcribed
            on_response: Callback when assistant response is ready
        """
        self.session_id = session_id or self._generate_session_id()
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response

        self._state = PipelineState.IDLE
        self._running = False
        self._lock = asyncio.Lock()

        # Components (initialized lazily)
        self._llm_client = None
        self._tts_client = None
        self._redis_client = None
        self._vector_store = None

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if the orchestrator is running."""
        return self._running

    def _set_state(self, state: PipelineState) -> None:
        """Update pipeline state and notify callback."""
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"session-{uuid.uuid4().hex[:8]}"

    async def initialize(self) -> None:
        """
        Initialize all pipeline components.

        Call this before running the pipeline.
        """
        pass  # Components will be initialized in subsequent commits

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        self._running = False
        self._set_state(PipelineState.IDLE)

    async def process_text(self, text: str) -> PipelineResult:
        """
        Process text input through the pipeline (skip STT).

        Useful for testing or text-based interaction.

        Args:
            text: User input text

        Returns:
            Pipeline result with response and metadata
        """
        pass  # Will be implemented in subsequent commits

    async def process_audio(self, audio_data: bytes) -> PipelineResult:
        """
        Process audio input through the full pipeline.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Pipeline result with response and metadata
        """
        pass  # Will be implemented in subsequent commits

    async def run_interactive(self) -> None:
        """
        Run the orchestrator in interactive mode.

        Continuously listens for voice input and responds.
        Press Ctrl+C to stop.
        """
        pass  # Will be implemented in subsequent commits
