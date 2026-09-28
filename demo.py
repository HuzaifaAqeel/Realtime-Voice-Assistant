"""Scripted demo mode: walks the full STT -> LLM -> TTS pipeline without a mic.

Usage:
    python demo.py [--audio sample_audio/hello.wav]

Every stage does real local processing. Stages that need a paid API key use a
clearly-labeled stand-in when the key is absent:

  STAGE 1 CAPTURE  load the sample WAV, real audio analysis (duration, RMS,
                   voice-activity segments)
  STAGE 2 STT      Groq Whisper if GROQ_API_KEY is set, else the scripted
                   transcript of the known sample file ("demo transcript")
  STAGE 3 LLM      Groq Llama if GROQ_API_KEY is set, else a local
                   keyword-based responder ("demo responder")
  STAGE 4 TTS      ElevenLabs if ELEVENLABS_API_KEY is set, else local
                   tone-synthesized WAV (fully offline). Set DEMO_GTTS=1
                   to opt into keyless gTTS instead.
                   The reply audio is saved to demo_reply.<mp3|wav>.

With all keys set this is the real pipeline; without them it is an honest,
fully-local walkthrough of every stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CHANNELS,
    CHUNK,
    ELEVENLABS_API_KEY,
    GROQ_API_KEY,
    RATE,
    SILENCE_THRESHOLD,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_audio"


def banner(stage: str, title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {stage}: {title}")
    print("=" * 64)


# ---------------------------------------------------------------- stage 1

def analyze_audio(path: Path) -> dict:
    """Real local audio analysis: decode + measure + voice activity."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    duration = n_frames / framerate
    rms = float(np.sqrt(np.mean(samples ** 2)))

    # voice activity: per-chunk RMS vs the same threshold as the live VAD
    chunk_len = int(framerate * 0.1)
    speech_chunks = 0
    segments: list[tuple[float, float]] = []
    start = None
    for i in range(0, len(samples), chunk_len):
        chunk = samples[i : i + chunk_len]
        active = float(np.sqrt(np.mean(chunk ** 2))) >= SILENCE_THRESHOLD
        t = i / framerate
        if active:
            speech_chunks += 1
            if start is None:
                start = t
        elif start is not None:
            segments.append((start, t))
            start = None
    if start is not None:
        segments.append((start, duration))
    return {
        "duration": duration,
        "sample_rate": framerate,
        "channels": n_channels,
        "sampwidth": sampwidth,
        "rms": rms,
        "speech_fraction": speech_chunks / max(1, len(samples) // chunk_len),
        "segments": segments,
    }


# ---------------------------------------------------------------- stage 2

def transcribe(audio_path: Path) -> tuple[str, str]:
    """Returns (transcript, backend_label)."""
    if GROQ_API_KEY:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as fh:
            result = client.audio.transcriptions.create(
                file=(audio_path.name, fh.read()), model="whisper-large-v3"
            )
        return result.text, "groq whisper-large-v3 (live API)"
    transcript_file = audio_path.with_suffix(".txt")
    if transcript_file.exists():
        text = transcript_file.read_text(encoding="utf-8").strip()
    else:
        text = "(no transcript file found)"
    return text, "demo transcript (scripted -- set GROQ_API_KEY for live Whisper)"


# ---------------------------------------------------------------- stage 3

def local_responder(text: str) -> str:
    """Deterministic local stand-in for the LLM (demo mode only)."""
    lowered = text.lower()
    if any(w in lowered for w in ("hello", "hi", "hey")):
        return (
            "Hmm, hello there! I'm running in demo mode, so my brain is a "
            "tiny local script right now -- but the pipeline around me is real."
        )
    if "name" in lowered:
        return "Ohh, I'm the realtime voice assistant demo -- STT to LLM to TTS, end to end."
    if "weather" in lowered:
        return "I can't check the weather in demo mode -- no live LLM key. Try me with a real key!"
    return (
        f"Got it -- you said: {text[:80]}. "
        "In demo mode I echo with a scripted brain; add GROQ_API_KEY for real Llama replies."
    )


def respond(text: str) -> tuple[str, str]:
    if GROQ_API_KEY:
        from agent import Agent

        agent = Agent()
        chunks: list[str] = []
        for token in agent.chat(text, streaming=True):
            print(token, end="", flush=True)
            chunks.append(token)
        print()
        return "".join(chunks), "groq llama-3.3-70b (live API)"
    reply = local_responder(text)
    print(reply)
    return reply, "demo responder (local -- set GROQ_API_KEY for live Llama)"


# ---------------------------------------------------------------- stage 4

def synthesize_tone_wav(text: str, out_path: Path) -> None:
    """Local fallback voice: phrase-shaped tone clusters (clearly synthetic)."""
    words = text.split()
    sr = 22050
    parts: list[np.ndarray] = []
    rng = np.random.default_rng(7)
    for i, word in enumerate(words):
        dur = min(0.6, 0.12 + 0.02 * len(word))
        t = np.arange(int(sr * dur)) / sr
        freq = 180 + 40 * np.sin(i * 0.9) + 10 * rng.standard_normal()
        tone = np.sin(2 * np.pi * (220 + freq) * t) * np.exp(-3 * t)
        # crude syllable envelope: 2 amplitude bumps per word
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t / max(dur, 1e-3))
        parts.append((tone * env * 0.5))
        parts.append(np.zeros(int(sr * 0.05)))
    audio = np.concatenate(parts) if parts else np.zeros(sr)
    pcm = (audio / max(1e-9, np.abs(audio).max()) * 20000).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def speak(text: str, out_dir: Path) -> tuple[Path, str]:
    if ELEVENLABS_API_KEY:
        from voice_assistant import VoiceAssistant

        assistant = VoiceAssistant.__new__(VoiceAssistant)  # skip mic init
        from elevenlabs.client import ElevenLabs
        from config import VOICE_ID

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.text_to_speech.convert(
            voice_id=VOICE_ID.value,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
        )
        out = out_dir / "demo_reply.mp3"
        with open(out, "wb") as fh:
            for chunk in response:
                if chunk:
                    fh.write(chunk)
        return out, "elevenlabs eleven_multilingual_v2 (live API)"
    # keyless path: fully local tone synthesis by default (no network).
    # Opt into gTTS (keyless Google TTS, needs network) with DEMO_GTTS=1.
    if os.environ.get("DEMO_GTTS") == "1":
        try:
            from gtts import gTTS

            out = out_dir / "demo_reply.mp3"
            gTTS(text=text, lang="en").save(str(out))
            return out, "gTTS (keyless Google TTS -- set ELEVENLABS_API_KEY for ElevenLabs)"
        except Exception:
            pass  # fall through to the local synthesizer
    out = out_dir / "demo_reply.wav"
    synthesize_tone_wav(text, out)
    return out, "local tone synthesis (fully offline demo voice -- clearly synthetic)"


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted voice-assistant demo.")
    parser.add_argument("--audio", default=str(SAMPLE_DIR / "hello.wav"))
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    if not audio_path.exists():
        print(f"sample audio not found: {audio_path}")
        return 1

    print("Realtime Voice Assistant -- scripted demo")
    print(f"sample: {audio_path.name}")

    banner("STAGE 1", "CAPTURE -- local audio analysis")
    info = analyze_audio(audio_path)
    print(f"  duration      : {info['duration']:.2f}s")
    print(f"  sample rate   : {info['sample_rate']} Hz, {info['channels']} ch")
    print(f"  RMS level     : {info['rms']:.1f}")
    print(f"  speech content: {info['speech_fraction'] * 100:.0f}% of chunks")
    for s, e in info["segments"]:
        print(f"  speech segment: {s:.1f}s -> {e:.1f}s")

    banner("STAGE 2", "STT -- speech to text")
    transcript, backend = transcribe(audio_path)
    print(f"  backend: {backend}")
    print(f"  transcript: \"{transcript}\"")

    banner("STAGE 3", "LLM -- response generation")
    print(f"  user: \"{transcript}\"")
    print("  assistant: ", end="", flush=True)
    reply, backend = respond(transcript)
    print(f"  backend: {backend}")

    banner("STAGE 4", "TTS -- text to speech")
    out_path, backend = speak(reply, out_dir)
    size_kb = out_path.stat().st_size / 1024
    print(f"  backend: {backend}")
    print(f"  saved: {out_path} ({size_kb:.1f} KB)")
    print("\nDemo complete -- pipeline walked end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
