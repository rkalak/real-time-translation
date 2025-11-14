# Real-Time Voice Translation Middleware

A real-time simultaneous interpretation pipeline that translates English speech to Telugu using:
- **Deepgram** for speech-to-text (ASR)
- **OpenAI GPT-4o** for translation
- **ElevenLabs** for text-to-speech (TTS)

## Features

- **Simultaneous Interpretation**: Translation starts while you're still speaking
- **Interim Diffing**: Intelligently handles ASR interim results to avoid duplicate words
- **Context-Aware Translation**: Maintains conversation history for coherent translations
- **Low-Latency Streaming**: Audio plays as it's generated

## Setup

1. **Install dependencies**:
   ```bash
   cd real-time-voice-middleware
   python3 -m venv venv
   source venv/bin/activate
   pip install deepgram-sdk openai elevenlabs pyaudio
   ```

2. **Set environment variables**:
   ```bash
   export DEEPGRAM_API_KEY=your_deepgram_key
   export OPENAI_API_KEY=your_openai_key
   export ELEVENLABS_API_KEY=your_elevenlabs_key
   ```

3. **Run the pipeline**:
   ```bash
   python pipeline.py
   ```

## Architecture

The pipeline consists of three async workers:

1. **ASR Worker**: Captures microphone input, sends to Deepgram, implements interim diffing
2. **LLM Worker**: Receives text chunks, translates to Telugu using OpenAI, streams tokens
3. **TTS Worker**: Receives translation tokens, buffers them, sends to ElevenLabs, plays audio

## Configuration

- **Input Language**: English (US)
- **Output Language**: Telugu
- **Voice**: Alisha (Indian female voice, optimized for Telugu)
- **Audio Format**: PCM 16kHz, 16-bit, mono

## Notes

- The system uses simultaneous interpretation, which trades a small amount of translation quality for significantly reduced latency
- Interim ASR results are intelligently diffed to avoid sending duplicate text to the translator
- The TTS worker buffers tokens and sends them in chunks to balance latency and audio quality

