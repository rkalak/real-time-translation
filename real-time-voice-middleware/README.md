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
│   ├── asr_worker.py        # Automatic speech recognition
│   ├── config.py            # Parameter adjustment
│   ├── latency_tracker.py   # Latency measurement and reporting
│   ├── pipeline.py          # Main translation pipeline
│   ├── requirements.txt     # Python dependencies
│   └── README.md             # Detailed documentation
```

## Current Configuration

- Input Language: English (US)
- Output Language: Spanish
- Voice: Molete (Spanish female voice)
- Audio Format: PCM 16kHz, 16-bit, mono

## Dependencies

The pipeline requires the following Python packages:

- **deepgram-sdk** (>=5.3.0) - For speech-to-text recognition
- **openai** (>=1.0.0) - For translation using GPT models
- **elevenlabs** (>=2.0.0) - For text-to-speech synthesis
- **pyaudio** (>=0.2.14) - For microphone input and audio playback

### Installing Dependencies

**Option 1: Using requirements.txt (Recommended)**

```bash
cd real-time-voice-middleware
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install deepgram-sdk openai elevenlabs pyaudio
```

**Note for PyAudio**: On some systems, PyAudio may require additional system dependencies:
- **macOS**: `brew install portaudio`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Windows**: Usually installs without additional dependencies

## Quick Start / Usage Instructions

### 1. Navigate to the Project Directory

```bash
cd real-time-voice-middleware
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

The pipeline requires three API keys. Set them as environment variables using the `export` command:

```bash
export DEEPGRAM_API_KEY=your_deepgram_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
export ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

**To make API keys persistent** (recommended), add them to your shell profile:

**For zsh (macOS default):**
```bash
echo 'export DEEPGRAM_API_KEY=your_deepgram_api_key_here' >> ~/.zshrc
echo 'export OPENAI_API_KEY=your_openai_api_key_here' >> ~/.zshrc
echo 'export ELEVENLABS_API_KEY=your_elevenlabs_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

**For bash:**
```bash
echo 'export DEEPGRAM_API_KEY=your_deepgram_api_key_here' >> ~/.bashrc
echo 'export OPENAI_API_KEY=your_openai_api_key_here' >> ~/.bashrc
echo 'export ELEVENLABS_API_KEY=your_elevenlabs_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

### 4. Activate Virtual Environment (Optional)

If you're using a virtual environment:

```bash
source venv/bin/activate
```

### 5. Run the Pipeline

```bash
python pipeline.py
```

The pipeline will:
- Check that all API keys are set
- Start the ASR, LLM, and TTS workers
- Open your microphone for input
- Begin translating your speech in real-time

