"""Configuration: everything secret comes from the environment.

Required:
    GROQ_API_KEY        Groq console (Whisper STT + Llama)  https://console.groq.com/keys
    ELEVENLABS_API_KEY  ElevenLabs (text-to-speech)         https://elevenlabs.io/

Optional:
    OPENAI_API_KEY      legacy OpenAI Whisper path (speech_to_text); the Groq
                        Whisper path (speech_to_text_groqs) is the default.
    VOICE_ID            one of ADAM / APHRODITE / CJ_MURPH (default ADAM)

Copy .env.example to .env and fill in your keys. Never commit .env.
"""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def require_keys(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". See .env.example."
        )


class Voices(Enum):
    APHRODITE = "fQuiOHUGZu5WDKWT80Wz"
    ADAM = "pNInz6obpgDQGcFmaJgB"
    CJ_MURPH = "876MHA6EtWKaHTEGzjy5"


def _voice_from_env() -> Voices:
    name = os.environ.get("VOICE_ID", "ADAM").upper()
    return Voices[name] if name in Voices.__members__ else Voices.ADAM


VOICE_ID = _voice_from_env()

# --- audio pipeline constants ---
FORMAT_WIDTH = 2          # 16-bit samples
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 200   # RMS below this counts as silence
SILENCE_DURATION = 2      # seconds of silence to stop recording
PRE_SPEECH_BUFFER_DURATION = 0.5  # seconds of audio kept before speech onset
