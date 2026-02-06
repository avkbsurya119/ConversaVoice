# ConversaVoice

Context-aware voice assistant with emotional intelligence, powered by Distil-Whisper, Llama 3, and Redis.

## Current Features

- **Real-time transcription** - Live microphone input with voice activity detection (Distil-Whisper)
- **File transcription** - Support for audio files with automatic chunking
- **GPU accelerated** - Optimized for NVIDIA GPUs (CUDA)
- **Timestamp support** - Get word-level timestamps for transcriptions
- **Intelligent responses** - Groq API with Llama 3 for emotionally-aware replies
- **Conversation memory** - Redis-based session storage and context persistence
- **Repetition detection** - Cosine similarity to detect when users repeat themselves

## Usage

```bash
# Activate environment
.\venv\Scripts\activate

# Real-time microphone transcription
python transcribe.py --realtime

# Continuous transcription mode
python transcribe.py --continuous

# Transcribe audio file
python transcribe.py audio.wav

# With timestamps
python transcribe.py audio.wav --timestamps
```

## LLM Usage (Groq)

```python
from src.llm import GroqClient

client = GroqClient()
response = client.get_emotional_response("Where is my order?")

print(response.reply)   # "I understand your concern..."
print(response.style)   # "empathetic"
print(response.pitch)   # "-5%"
print(response.rate)    # "0.85"
```

## Memory Usage (Redis)

```python
from src.memory import RedisClient, VectorStore

# Session management
redis = RedisClient()
redis.create_session("user-123")
redis.add_message("user-123", "user", "Hello!")
context = redis.get_context_string("user-123")

# Repetition detection
vectors = VectorStore(redis)
result = vectors.check_repetition("user-123", "Where is my order?")
if result.is_repetition:
    print(f"User is repeating (similarity: {result.score:.0%})")
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- FFmpeg (for audio file processing)
- Redis server (local or Docker)
- Groq API key (free at console.groq.com)

## Installation

```bash
# Clone and setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Start Redis (Docker)
docker run -d --name redis -p 6379:6379 redis

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Architecture (Conversia/Zudu.ai)

Building a context-aware voice assistant with emotional intelligence:

### Phase 1: Brain (LLM Integration) - COMPLETE
- Groq API with Llama 3 for intelligent responses
- Context-aware conversation handling
- Emotion detection with prosody parameters (style, pitch, rate)

### Phase 2: Memory (Redis) - COMPLETE
- Session-based conversation history
- Vector similarity for repetition detection (threshold: 0.85)
- Context retrieval for coherent responses

### Phase 3: Voice Output (TTS) - PENDING
- Azure Neural TTS integration
- SSML-based emotional prosody control
- Streaming audio for low latency

### Phase 4: Orchestrator - PENDING
- Async pipeline: Mic -> Whisper -> LLM -> TTS -> Speaker
- Sub-second response times
- Real-time emotion adaptation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Distil-Whisper Large-v3-Turbo |
| LLM | Groq API (Llama 3) |
| Memory | Redis |
| Text-to-Speech | Azure Neural TTS |
| GPU | CUDA (RTX 4060) |

## License

MIT
