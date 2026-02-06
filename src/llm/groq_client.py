"""
Groq API client for ConversaVoice.
Provides async interface to Llama 3 for intelligent, emotionally-aware responses.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class GroqConfig:
    """Configuration for Groq API client."""
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024


class GroqClient:
    """
    Async client for Groq API with Llama 3.

    Provides emotionally intelligent responses with prosody parameters
    for text-to-speech synthesis.
    """

    def __init__(self, config: Optional[GroqConfig] = None):
        """
        Initialize the Groq client.

        Args:
            config: Optional GroqConfig. If not provided, uses environment variables.
        """
        if config is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            config = GroqConfig(api_key=api_key)

        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.config.api_key)
        return self._client

    def chat(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Send a message to Llama 3 and get a response.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            The assistant's response text.
        """
        client = self._get_client()

        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })

        # Add context if provided
        if context:
            messages.append({
                "role": "user",
                "content": f"Previous context: {context}"
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Call Groq API
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.choices[0].message.content

    def _get_system_prompt(self) -> str:
        """Get the system prompt. To be implemented in next commit."""
        return "You are a helpful assistant."
