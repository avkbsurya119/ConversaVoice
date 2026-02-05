# ConversaVoice

Local real-time speech transcription powered by Distil-Whisper Large-v3-Turbo with GPU acceleration.

## Current Features

- **Real-time transcription** - Live microphone input with voice activity detection
- **File transcription** - Support for audio files with automatic chunking
- **GPU accelerated** - Optimized for NVIDIA GPUs (CUDA)
- **Timestamp support** - Get word-level timestamps for transcriptions

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

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- FFmpeg (for audio file processing)

## Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Future Plans (Conversia/Zudu.ai Architecture)

Building toward a context-aware voice assistant with emotional intelligence:

### Phase 1: Brain (LLM Integration)
- Groq API with Llama 3 for intelligent responses
- Context-aware conversation handling
- Emotion detection from user input

### Phase 2: Memory (Redis)
- Session-based conversation history
- Vector similarity for repetition detection
- Context retrieval for coherent responses

### Phase 3: Voice Output (TTS)
- Azure Neural TTS integration
- SSML-based emotional prosody control
- Streaming audio for low latency

### Phase 4: Orchestrator
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
