# Realtime-Voice-Assistant

A **realtime voice-to-voice assistant** running the full pipeline:

```
 microphone ──▶ STT ──▶ LLM ──▶ TTS ──▶ speaker
              Whisper   Llama   ElevenLabs
              (Groq)    (Groq)
```

- **STT** — Groq's `whisper-large-v3`: fast, accurate speech-to-text.
- **LLM** — Groq's `llama-3.3-70b-versatile` with rolling conversation memory,
  prompted for short, natural spoken-style replies.
- **TTS** — ElevenLabs `eleven_multilingual_v2` with streaming playback.
- **VAD** — RMS-based voice activity detection: recording starts on speech and
  stops after 2 seconds of silence, with a 0.5s pre-speech buffer so the first
  syllable is never clipped.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY + ELEVENLABS_API_KEY

# live mode (needs a microphone; pyaudio requires portaudio system-wide)
python voice_assistant.py

# scripted demo mode (no mic, no keys needed)
python demo.py
```

Speak into your mic — the assistant transcribes, thinks, and answers out loud.

## Demo mode

`python demo.py` walks the **entire pipeline** on the included sample audio
(`sample_audio/hello.wav`, real spoken speech) with real local processing at
every stage:

1. **Capture** — decodes the WAV and reports duration, RMS level, and
   voice-activity segments using the same VAD as live mode.
2. **STT** — live Groq Whisper when `GROQ_API_KEY` is set, otherwise the
   scripted transcript of the known sample (clearly labeled).
3. **LLM** — live Llama when the key is set, otherwise a local keyword
   responder (clearly labeled).
4. **TTS** — live ElevenLabs when `ELEVENLABS_API_KEY` is set, otherwise
   keyless gTTS, saving `demo_reply.mp3`.

## Project structure

```
├── voice_assistant.py  # mic loop: VAD -> STT -> LLM -> TTS -> playback
├── agent.py            # LangChain chat agent (Groq Llama + memory)
├── config.py           # env-var config (keys never hardcoded)
├── demo.py             # scripted end-to-end demo (no mic/keys needed)
└── sample_audio/       # hello.wav + hello.txt (known sample + transcript)
```

## Notes

- All secrets come from environment variables — see `.env.example`.
  Nothing is hardcoded, and `.env` is git-ignored.
- `pyaudio` is only needed for live microphone mode; demo mode, config, and
  the agent import fine without it.
- Tested with Python 3.11+.

## License

MIT — see [LICENSE](LICENSE).
