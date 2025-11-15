# StarLife - Real-Time Voice Translation

A real-time simultaneous interpretation pipeline that translates English speech to multiple target languages using advanced AI services. Currently working with English to Spanish

## Features

- Translation starts while you're still speaking
- Handles ASR interim results to avoid duplicate words
- Maintains conversation history for coherence
- Audio plays in chunks as it's generated
- Currently configured for Spanish

## Technology Stack

- Deepgram: Speech-to-text (ASR) with medical model support
- OpenAI GPT-4o-mini: Real-time translation with context awareness
- ElevenLabs: TTS

## Project Structure

```
starlife/
├── real-time-voice-middleware/
│   ├──asr_worker.py        # automatic speech recognition
│   ├──config.py            # parameter adjustment
│   ├──latency_tracker.py   # 
│   ├──pipeline.py          # Main translation pipeline
│   ├──pipeline.py          # Main translation pipeline
│   └── README.md            # Detailed documentation
```

## Current Configuration

- Input Language: English (US)
- Output Language: Spanish
- Voice: Molete (Spanish female voice)
- Audio Format: PCM 16kHz, 16-bit, mono

## License

