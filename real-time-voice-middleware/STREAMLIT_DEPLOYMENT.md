# Streamlit Web App Deployment Guide

This guide explains how to deploy the real-time voice translation pipeline as a web app using Streamlit.

## Overview

The Streamlit app (`app.py`) provides a web-based interface for the translation pipeline. Users can:
- Record audio directly in their browser
- See real-time transcriptions and translations
- Hear Spanish audio output automatically
- No need to install dependencies or set up API keys (you handle that)

## Local Development

### 1. Install Dependencies

```bash
cd real-time-voice-middleware
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.streamlit/secrets.toml` file (copy from `.streamlit/secrets.toml.example`):

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your API keys:

```toml
DEEPGRAM_API_KEY = "your_deepgram_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment to Streamlit Community Cloud

### Step 1: Push Your Code to GitHub

1. Create a GitHub repository (if you haven't already)
2. Push your code:

```bash
git add .
git commit -m "Add Streamlit web app"
git push origin main
```

### Step 2: Deploy to Streamlit Community Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to: `real-time-voice-middleware/app.py`
6. Click "Deploy"

### Step 3: Configure API Keys (Secrets)

1. In your Streamlit app dashboard, go to "Settings" → "Secrets"
2. Add your API keys:

```toml
DEEPGRAM_API_KEY = "your_deepgram_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
```

3. Save and the app will automatically redeploy

### Step 4: Share Your App

Once deployed, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

Share this URL with anyone who wants to use your translation app!

## Deployment to Hugging Face Spaces

### Step 1: Create a Hugging Face Space

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as the SDK
4. Name your space (e.g., `starlife-voice-translation`)

### Step 2: Upload Your Code

1. Clone the space:
```bash
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name
```

2. Copy your files:
```bash
cp ../real-time-voice-middleware/app.py .
cp ../real-time-voice-middleware/config.py .
cp ../real-time-voice-middleware/requirements.txt .
```

3. Create a `.streamlit/secrets.toml` file (or use HF Secrets)

4. Commit and push:
```bash
git add .
git commit -m "Initial deployment"
git push
```

### Step 3: Configure Secrets

1. In your Space settings, go to "Variables and secrets"
2. Add your API keys as secrets:
   - `DEEPGRAM_API_KEY`
   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`

## How It Works

### Differences from `pipeline.py`

The Streamlit app (`app.py`) adapts the pipeline for web deployment:

1. **Audio Input**: Uses Streamlit's `st.audio_input` widget instead of PyAudio microphone
2. **Audio Output**: Uses Streamlit's `st.audio` widget for browser playback instead of PyAudio speakers
3. **API Keys**: Uses Streamlit secrets instead of environment variables
4. **UI**: Provides a web interface with real-time updates
5. **Processing**: Processes audio chunks on-demand rather than continuous streaming

### Architecture

```
Browser Audio Input → Streamlit App → Deepgram ASR → OpenAI Translation → ElevenLabs TTS → Browser Audio Output
```

## Troubleshooting

### "API Keys Missing" Error

- **Local**: Make sure `.streamlit/secrets.toml` exists and has all three keys
- **Deployed**: Check that secrets are configured in Streamlit Community Cloud or Hugging Face Spaces settings

### Audio Not Playing

- Check browser console for errors
- Ensure audio format is supported (WAV/PCM)
- Try a different browser (Chrome/Firefox recommended)

### Processing Errors

- Check that all API keys are valid
- Verify API quotas haven't been exceeded
- Check Streamlit logs for detailed error messages

### Slow Processing

- This is normal - each audio chunk goes through ASR → Translation → TTS
- Typical latency: 2-5 seconds per chunk
- For faster processing, reduce audio chunk size or adjust model settings

## Security Notes

⚠️ **Important**: Never commit your actual API keys to GitHub!

- Use `.streamlit/secrets.toml` for local development (and add it to `.gitignore`)
- Use Streamlit Community Cloud or Hugging Face Spaces secrets for deployed apps
- Never share your API keys publicly

## Cost Considerations

The web app uses the same APIs as the local pipeline:
- **Deepgram**: Pay-per-minute of audio transcribed
- **OpenAI**: Pay-per-token for translation
- **ElevenLabs**: Pay-per-character for TTS

Monitor your usage in each service's dashboard to avoid unexpected charges.

## Next Steps

- Customize the UI in `app.py`
- Add more languages by modifying the system prompt
- Add user authentication if needed
- Implement rate limiting for public deployments

