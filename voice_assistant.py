"""Realtime voice assistant: microphone -> Whisper STT -> Llama -> ElevenLabs TTS.

pyaudio is imported lazily so the module (and demo mode) works on machines
without audio hardware or portaudio installed.
"""

import wave
from io import BytesIO
from time import time
from typing import Iterator, Optional

import numpy as np
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from groq import Groq
from pydub import AudioSegment

from agent import Agent
from config import (
    CHANNELS,
    CHUNK,
    CHUNK as _CHUNK,  # noqa: F401  (re-export for tests)
    ELEVENLABS_API_KEY,
    FORMAT_WIDTH,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    PRE_SPEECH_BUFFER_DURATION,
    RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
    VOICE_ID,
    require_keys,
)


def _pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "pyaudio is required for live microphone mode. Install portaudio "
            "system-wide, then `pip install pyaudio`. Demo mode (--demo) "
            "does not need it."
        ) from exc
    return pyaudio


class VoiceAssistant:
    def __init__(self, voice_id: Optional[str] = None):
        require_keys("GROQ_API_KEY", "ELEVENLABS_API_KEY")
        self.voice_id = voice_id or VOICE_ID.value
        self.agent = Agent()
        self.xi_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.g_client = Groq(api_key=GROQ_API_KEY)
        self.oai_client = None
        if OPENAI_API_KEY:
            from openai import OpenAI

            self.oai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio = None  # created lazily in live mode

    # ---------------- live microphone ----------------

    def _ensure_audio(self):
        if self.audio is None:
            self.audio = _pyaudio().PyAudio()
        return self.audio

    def is_silence(self, data: bytes) -> bool:
        """RMS-based voice activity detection: True when the chunk is silence."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        if audio_data.size == 0:
            return True
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
        return rms < SILENCE_THRESHOLD

    def listen_for_speech(self) -> BytesIO:
        """Block until speech starts, then record until trailing silence."""
        pa = _pyaudio()
        audio = self._ensure_audio()
        stream = audio.open(
            format=pa.paInt16, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        print("Listening for speech...")
        pre_speech_buffer: list[bytes] = []
        pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            pre_speech_buffer.append(data)
            if len(pre_speech_buffer) > pre_speech_chunks:
                pre_speech_buffer.pop(0)
            if not self.is_silence(data):
                print("Speech detected, start recording...")
                stream.stop_stream()
                stream.close()
                return self.record_audio(pre_speech_buffer)

    def record_audio(self, pre_speech_buffer: list[bytes]) -> BytesIO:
        pa = _pyaudio()
        audio = self._ensure_audio()
        stream = audio.open(
            format=pa.paInt16, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        frames = list(pre_speech_buffer)
        silent_chunks = 0
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            silent_chunks = silent_chunks + 1 if self.is_silence(data) else 0
            if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                break
        stream.stop_stream()
        stream.close()

        audio_bytes = BytesIO()
        with wave.open(audio_bytes, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(FORMAT_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        audio_bytes.seek(0)
        return audio_bytes

    # ---------------- pipeline stages ----------------

    def speech_to_text(self, audio_bytes: BytesIO) -> str:
        """Transcribe with OpenAI Whisper (legacy path; needs OPENAI_API_KEY)."""
        if self.oai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not set; use speech_to_text_groq.")
        audio_bytes.seek(0)
        transcription = self.oai_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-1",
        )
        return transcription.text

    def speech_to_text_groq(self, audio_bytes: BytesIO) -> str:
        """Transcribe with Groq Whisper (default, fast)."""
        start = time()
        audio_bytes.seek(0)
        transcription = self.g_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-large-v3",
        )
        print(f"Transcription took {time() - start:.2f}s")
        return transcription.text

    def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> BytesIO:
        """Synthesize speech with ElevenLabs; returns MP3 bytes."""
        response = self.xi_client.text_to_speech.convert(
            voice_id=voice_id or self.voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        )
        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)
        return audio_stream

    def audio_stream_to_iterator(
        self, audio_stream: BytesIO, format: str = "mp3"
    ) -> Iterator[bytes]:
        audio = AudioSegment.from_file(audio_stream, format=format)
        audio = audio.set_frame_rate(22050).set_channels(2).set_sample_width(2)
        raw_data = audio.raw_data
        chunk_size = 1024
        for i in range(0, len(raw_data), chunk_size):
            yield raw_data[i : i + chunk_size]

    def stream_audio(
        self, audio_bytes_iterator: Iterator[bytes], rate: int = 22050,
        channels: int = 2,
    ) -> None:
        pa = _pyaudio()
        audio = self._ensure_audio()
        stream = audio.open(format=pa.paInt16, channels=channels, rate=rate, output=True)
        try:
            for audio_chunk in audio_bytes_iterator:
                stream.write(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def chat(self, query: str) -> str:
        start = time()
        response = self.agent.chat(query)
        print(f"Response: {response}\nResponse Time: {time() - start:.2f}s")
        return response

    # ---------------- main loop ----------------

    def run(self) -> None:
        """Live loop: listen -> transcribe (Groq Whisper) -> Llama -> speak."""
        while True:
            audio_bytes = self.listen_for_speech()
            text = self.speech_to_text_groq(audio_bytes)
            print(f"Heard: {text}")
            response_text = self.chat(text)
            audio_stream = self.text_to_speech(response_text)
            self.stream_audio(self.audio_stream_to_iterator(audio_stream))


if __name__ == "__main__":
    VoiceAssistant().run()
