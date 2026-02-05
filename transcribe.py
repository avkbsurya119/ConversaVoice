import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import sys
import os

# Set HF token from environment variable (optional, for faster downloads)
# Set HF_TOKEN environment variable if you have one

class WhisperTranscriber:
    def __init__(self, model_id="distil-whisper/distil-large-v3"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        self.pipe = None
        self.sample_rate = 16000

    def load_model(self):
        """Load the Whisper model."""
        if self.pipe is not None:
            return self.pipe

        print(f"Loading model on {self.device}...")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.dtype,
            device=self.device,
        )

        print("Model loaded successfully!")
        return self.pipe

    def transcribe_file(self, audio_path, chunk_length_s=30, batch_size=8):
        """Transcribe an audio file with chunking support for long audio."""
        self.load_model()

        result = self.pipe(
            audio_path,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True
        )

        return result

    def transcribe_audio(self, audio_array, sample_rate=16000):
        """Transcribe a numpy audio array."""
        self.load_model()

        # Resample if needed
        if sample_rate != self.sample_rate:
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio_array).float()
            audio_array = F.resample(audio_tensor, sample_rate, self.sample_rate).numpy()

        result = self.pipe({"array": audio_array, "sampling_rate": self.sample_rate})
        return result["text"]

    def transcribe_batch(self, audio_paths, chunk_length_s=30, batch_size=8):
        """Transcribe multiple audio files."""
        self.load_model()
        results = {}

        for path in audio_paths:
            print(f"Transcribing: {path}")
            results[path] = self.transcribe_file(path, chunk_length_s, batch_size)

        return results


class RealtimeTranscriber:
    def __init__(self, transcriber=None):
        self.transcriber = transcriber or WhisperTranscriber()
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds per chunk
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5  # minimum speech to transcribe

    def _is_speech(self, audio_chunk):
        """Simple voice activity detection."""
        return np.abs(audio_chunk).mean() > self.silence_threshold

    def start_microphone(self, callback=None):
        """Start real-time transcription from microphone."""
        import pyaudio

        self.transcriber.load_model()

        p = pyaudio.PyAudio()

        # Find default input device
        try:
            default_device = p.get_default_input_device_info()
            print(f"Using microphone: {default_device['name']}")
        except:
            print("No microphone found!")
            return

        chunk_samples = int(self.sample_rate * self.chunk_duration)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        print("\n[Listening... Press Ctrl+C to stop]\n")

        audio_buffer = []

        try:
            while True:
                # Read audio chunk
                frames = []
                for _ in range(0, int(self.sample_rate / 1024 * self.chunk_duration)):
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))

                audio_chunk = np.concatenate(frames)

                # Check for speech
                if self._is_speech(audio_chunk):
                    audio_buffer.append(audio_chunk)
                elif audio_buffer:
                    # Process accumulated speech
                    full_audio = np.concatenate(audio_buffer)

                    if len(full_audio) / self.sample_rate >= self.min_speech_duration:
                        text = self.transcriber.transcribe_audio(full_audio, self.sample_rate)
                        if text.strip():
                            if callback:
                                callback(text)
                            else:
                                print(f"> {text}")

                    audio_buffer = []

        except KeyboardInterrupt:
            print("\n[Stopped]")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def transcribe_continuous(self, duration=None, callback=None):
        """Continuously transcribe for a specified duration or until stopped."""
        import pyaudio
        import time

        self.transcriber.load_model()

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        print("\n[Recording... Press Ctrl+C to stop]\n")

        start_time = time.time()
        all_text = []

        try:
            while True:
                if duration and (time.time() - start_time) >= duration:
                    break

                # Read 5 seconds of audio
                frames = []
                for _ in range(0, int(self.sample_rate / 1024 * self.chunk_duration)):
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))

                audio_chunk = np.concatenate(frames)

                if self._is_speech(audio_chunk):
                    text = self.transcriber.transcribe_audio(audio_chunk, self.sample_rate)
                    if text.strip():
                        all_text.append(text)
                        if callback:
                            callback(text)
                        else:
                            print(f"> {text}")

        except KeyboardInterrupt:
            print("\n[Stopped]")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        return " ".join(all_text)


# Convenience functions
_transcriber = None

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber

def load_model():
    """Load the model and return the pipeline."""
    return get_transcriber().load_model()

def transcribe(audio_path_or_array, sample_rate=16000):
    """Transcribe audio file or numpy array."""
    t = get_transcriber()
    if isinstance(audio_path_or_array, str):
        result = t.transcribe_file(audio_path_or_array)
        return result["text"]
    else:
        return t.transcribe_audio(audio_path_or_array, sample_rate)

def transcribe_realtime(callback=None):
    """Start real-time microphone transcription."""
    rt = RealtimeTranscriber(get_transcriber())
    rt.start_microphone(callback)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distil-Whisper Transcription")
    parser.add_argument("audio", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--realtime", "-r", action="store_true", help="Real-time microphone transcription")
    parser.add_argument("--continuous", "-c", action="store_true", help="Continuous transcription mode")
    parser.add_argument("--duration", "-d", type=float, help="Recording duration in seconds")
    parser.add_argument("--timestamps", "-t", action="store_true", help="Show timestamps")

    args = parser.parse_args()

    if args.realtime or args.continuous:
        rt = RealtimeTranscriber()
        if args.continuous:
            rt.transcribe_continuous(duration=args.duration)
        else:
            rt.start_microphone()
    elif args.audio:
        t = WhisperTranscriber()
        result = t.transcribe_file(args.audio)

        if args.timestamps and "chunks" in result:
            for chunk in result["chunks"]:
                start, end = chunk["timestamp"]
                print(f"[{start:.2f}s - {end:.2f}s] {chunk['text']}")
        else:
            print(result["text"])
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python transcribe.py audio.wav              # Transcribe file")
        print("  python transcribe.py audio.wav --timestamps # With timestamps")
        print("  python transcribe.py --realtime             # Real-time mic input")
        print("  python transcribe.py --continuous -d 60     # Record for 60 seconds")


if __name__ == "__main__":
    main()
