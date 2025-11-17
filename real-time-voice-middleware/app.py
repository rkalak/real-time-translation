"""
Streamlit Web App for Real-Time Voice Translation
Deploy this to Streamlit Community Cloud or Hugging Face Spaces
"""

import streamlit as st
import asyncio
import time
import io
import wave
from typing import Optional, List, Dict, Tuple
import queue
import threading

# Import pipeline components
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Language mappings
LANGUAGE_CODES = {
    "English": "en-US",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese": "pt-PT",
    "Chinese": "zh-CN",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Arabic": "ar-SA",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
}

# Default system prompt template
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a real-time, simultaneous medical interpreter translating {input_lang} to {output_lang}.
You will receive small, incremental chunks of {input_lang} text.
You must translate these chunks into {output_lang} IMMEDIATELY as you receive them.

CRITICAL RULES:
1. **SIMULTANEOUS:** Do NOT wait for the full sentence. Start translating the first chunk you get.
2. **INCREMENTAL:** Translate each new chunk and continue the previous translation naturally.
3. **OUTPUT {output_lang} ONLY:** Your response must be ONLY {output_lang} text - no explanations, no meta-commentary.
4. **CHOOSE ONE TRANSLATION:** When multiple translations are possible, choose the MOST LIKELY option.
5. **EXACT TRANSLATION:** Translate EXACTLY what is in the input. Do NOT add extra words or context.
6. **TRANSLATE ALL CHUNKS:** Every input chunk must be translated."""

# Try to import from config, but use defaults if not available
try:
    from config import (
        RATE, CHANNELS,
        ASR_BUFFER_DELAY, ASR_MIN_CHUNK_SIZE, ASR_MAX_CHUNK_SIZE,
        LLM_AUTOCORRECT_ENABLED, LLM_PREDICTION_ENABLED, LLM_PREDICTION_WINDOW,
        LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_MODEL,
        TTS_BUFFER_DELAY, TTS_MAX_BUFFER_LENGTH, TTS_MIN_CHUNK_LENGTH,
        TTS_SPACE_SEND_LENGTH, TTS_PUNCTUATION_WAIT,
        MEASURE_LATENCY, LATENCY_REPORT_INTERVAL
    )
except ImportError:
    # Fallback defaults
    RATE = 16000
    CHANNELS = 1
    ASR_BUFFER_DELAY = 0.0
    ASR_MIN_CHUNK_SIZE = 1
    ASR_MAX_CHUNK_SIZE = 5
    LLM_AUTOCORRECT_ENABLED = True
    LLM_PREDICTION_ENABLED = True
    LLM_PREDICTION_WINDOW = 3
    LLM_TEMPERATURE = 0.3
    LLM_MAX_RETRIES = 2
    LLM_MODEL = "gpt-4o-mini"
    TTS_BUFFER_DELAY = 0.05
    TTS_MAX_BUFFER_LENGTH = 75
    TTS_MIN_CHUNK_LENGTH = 20
    TTS_SPACE_SEND_LENGTH = 15
    TTS_PUNCTUATION_WAIT = True
    MEASURE_LATENCY = True
    LATENCY_REPORT_INTERVAL = 3

# Get API keys from Streamlit secrets or environment
def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.environ.get(key_name)

DEEPGRAM_API_KEY = get_api_key("DEEPGRAM_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
ELEVENLABS_API_KEY = get_api_key("ELEVENLABS_API_KEY")

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'transcription_history': [],
        'translation_history': [],
        'audio_outputs': [],
        'is_recording': False,
        'llm_messages': [],
        'text_buffer': "",
        'voice_id': "T4Au24Lt2uWk24Qra0No",
        'input_language': "English",
        'output_language': "Spanish",
        'last_audio_hash': None,
        'current_transcript': "",
        'current_translation': "",
        'tts_buffer': "",
        'processing_task': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Initialize clients
@st.cache_resource
def get_clients():
    """Initialize and cache API clients."""
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
    deepgram_client = AsyncDeepgramClient() if DEEPGRAM_API_KEY else None
    return openai_client, elevenlabs_client, deepgram_client

# Page configuration
st.set_page_config(
    page_title="StarLife - Real-Time Voice Translation",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("🌐 StarLife - Real-Time Voice Translation")
st.markdown("**Multi-Language Medical Interpreter**")

# Check API keys
if not all([DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY]):
    st.error("⚠️ **API Keys Missing**")
    st.markdown("""
    Please configure your API keys. For deployed apps, add them to Streamlit secrets.
    For local development, set them as environment variables or in `.streamlit/secrets.toml`.
    
    Required keys:
    - `DEEPGRAM_API_KEY`
    - `OPENAI_API_KEY`
    - `ELEVENLABS_API_KEY`
    """)
    st.stop()

# Get clients
openai_client, elevenlabs_client, deepgram_client = get_clients()

# Custom CSS for button colors
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #28a745;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838;
    }
    .stop-button {
        background-color: #dc3545 !important;
        color: white !important;
    }
    .stop-button:hover {
        background-color: #c82333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for language selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    st.subheader("🌍 Language Selection")
    input_lang = st.selectbox(
        "Input Language",
        options=list(LANGUAGE_CODES.keys()),
        index=list(LANGUAGE_CODES.keys()).index(st.session_state.input_language) if st.session_state.input_language in LANGUAGE_CODES else 0
    )
    output_lang = st.selectbox(
        "Output Language",
        options=list(LANGUAGE_CODES.keys()),
        index=list(LANGUAGE_CODES.keys()).index(st.session_state.output_language) if st.session_state.output_language in LANGUAGE_CODES else 1
    )
    
    # Update session state if languages changed
    if input_lang != st.session_state.input_language or output_lang != st.session_state.output_language:
        st.session_state.input_language = input_lang
        st.session_state.output_language = output_lang
        # Reset conversation history when languages change
        st.session_state.llm_messages = []
        st.session_state.transcription_history = []
        st.session_state.translation_history = []
        st.session_state.audio_outputs = []
    
    st.markdown("---")
    st.markdown(f"**Model:** {LLM_MODEL}")
    st.markdown(f"**Temperature:** {LLM_TEMPERATURE}")
    st.markdown(f"**Autocorrect:** {'✅ Enabled' if LLM_AUTOCORRECT_ENABLED else '❌ Disabled'}")
    
    if st.button("🔄 Clear History"):
        st.session_state.transcription_history = []
        st.session_state.translation_history = []
        st.session_state.audio_outputs = []
        st.session_state.llm_messages = []
        st.session_state.current_transcript = ""
        st.session_state.current_translation = ""
        st.session_state.tts_buffer = ""
        st.rerun()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🎤 Input ({st.session_state.input_language})")
    
    # Start/Stop button with custom styling
    if st.session_state.is_recording:
        # Red stop button
        button_html = """
        <style>
            div[data-testid="stButton"] > button[kind="primary"] {
                background-color: #dc3545;
                color: white;
                border: none;
            }
            div[data-testid="stButton"] > button[kind="primary"]:hover {
                background-color: #c82333;
            }
        </style>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        if st.button("🛑 Stop Talking", type="primary", use_container_width=True):
            st.session_state.is_recording = False
            st.session_state.current_transcript = ""
            st.session_state.current_translation = ""
            st.session_state.tts_buffer = ""
            st.session_state.last_audio_hash = None
            st.rerun()
    else:
        # Green start button
        button_html = """
        <style>
            div[data-testid="stButton"] > button[kind="primary"] {
                background-color: #28a745;
                color: white;
                border: none;
            }
            div[data-testid="stButton"] > button[kind="primary"]:hover {
                background-color: #218838;
            }
        </style>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        if st.button("🎤 Start Talking", type="primary", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.llm_messages = []  # Reset conversation when starting
            st.session_state.current_transcript = ""
            st.session_state.current_translation = ""
            st.rerun()
    
    # Audio input widget (only show when recording)
    if st.session_state.is_recording:
        audio_input = st.audio_input("Recording...", type="wav")
        
        if audio_input:
            audio_bytes = audio_input.read()
            
            # Check if this is new audio
            current_hash = hash(audio_bytes)
            if current_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = current_hash
                # Process audio in background
                process_audio_async_wrapper(audio_bytes)
    
    # Live transcription display
    st.markdown("### 📝 Live Transcription")
    if st.session_state.current_transcript:
        st.markdown(f"**{st.session_state.current_transcript}**")
    else:
        st.markdown("*Waiting for input...*")
    
    # Transcription history
    if st.session_state.transcription_history:
        with st.expander("📜 Transcription History"):
            for i, transcript in enumerate(reversed(st.session_state.transcription_history[-10:])):
                st.markdown(f"**{len(st.session_state.transcription_history) - i}.** {transcript}")

with col2:
    st.subheader(f"🔊 Output ({st.session_state.output_language})")
    
    # Live translation display
    st.markdown("### 🌐 Live Translation")
    if st.session_state.current_translation:
        st.markdown(f"**{st.session_state.current_translation}**")
    else:
        st.markdown("*Waiting for translation...*")
    
    # Audio output (only play when there's a natural stop)
    if st.session_state.audio_outputs:
        latest_audio = st.session_state.audio_outputs[-1]
        st.audio(latest_audio, format="audio/wav", autoplay=True)
    
    # Translation history
    if st.session_state.translation_history:
        with st.expander("📜 Translation History"):
            for i, translation in enumerate(reversed(st.session_state.translation_history[-10:])):
                st.markdown(f"**{len(st.session_state.translation_history) - i}.** {translation}")

# Processing functions
def process_audio_async_wrapper(audio_bytes: bytes):
    """Wrapper to process audio asynchronously."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_audio_pipeline(audio_bytes))
        finally:
            loop.close()
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        import traceback
        st.code(traceback.format_exc())

async def process_audio_pipeline(audio_bytes: bytes):
    """Process audio through the full pipeline: ASR → LLM → TTS (with tokenization)."""
    if not st.session_state.is_recording:
        return
    
    try:
        # Step 1: ASR (Speech-to-Text)
        transcript = await asr_process(audio_bytes)
        if transcript and st.session_state.is_recording:  # Check again after ASR
            st.session_state.current_transcript = transcript
            st.session_state.transcription_history.append(transcript)
            
            # Step 2: Translation with streaming (tokenization)
            if st.session_state.is_recording:  # Check before translation
                await llm_process_streaming(transcript)
        
        # Only rerun if still recording
        if st.session_state.is_recording:
            st.rerun()
    except Exception as e:
        if st.session_state.is_recording:  # Only show error if still recording
            st.error(f"Error in pipeline: {e}")
            import traceback
            st.code(traceback.format_exc())

async def asr_process(audio_bytes: bytes) -> Optional[str]:
    """Process audio through Deepgram ASR."""
    if not deepgram_client:
        return None
    
    try:
        input_lang_code = LANGUAGE_CODES[st.session_state.input_language]
        
        async with deepgram_client.listen.v1.connect(
            model="nova-3-medical",
            language=input_lang_code,
            smart_format="true",
            interim_results="false",
            encoding="linear16",
            sample_rate=str(RATE),
            channels=str(CHANNELS),
        ) as dg_connection:
            transcript_text = ""
            
            async def on_message(result):
                nonlocal transcript_text
                if isinstance(result, ListenV1ResultsEvent):
                    if result.channel and result.channel.alternatives:
                        transcript = result.channel.alternatives[0].transcript
                        if transcript:
                            transcript_text = transcript
            
            dg_connection.on(EventType.MESSAGE, on_message)
            listen_task = asyncio.create_task(dg_connection.start_listening())
            
            try:
                # Send audio in chunks
                chunk_size = 1024
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await dg_connection.send_media(chunk)
                
                await asyncio.sleep(0.5)  # Wait for final results
                
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
            
            return transcript_text if transcript_text else None
            
    except Exception as e:
        st.error(f"ASR Error: {e}")
        return None

async def llm_process_streaming(text: str):
    """Process text through OpenAI translation with streaming and tokenization."""
    if not openai_client:
        return
    
    try:
        # Get system prompt for current languages
        system_prompt = DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
            input_lang=st.session_state.input_language,
            output_lang=st.session_state.output_language
        )
        
        # Initialize messages if empty
        if not st.session_state.llm_messages:
            st.session_state.llm_messages = [{"role": "system", "content": system_prompt}]
        else:
            # Update system prompt if languages changed
            st.session_state.llm_messages[0] = {"role": "system", "content": system_prompt}
        
        # Add user message
        st.session_state.llm_messages.append({"role": "user", "content": text})
        
        # Stream translation tokens
        translation_buffer = []
        tts_token_buffer = []  # Buffer for TTS tokens
        
        response_stream = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=st.session_state.llm_messages,
            stream=True,
            temperature=LLM_TEMPERATURE
        )
        
        async for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    token = delta.content
                    translation_buffer.append(token)
                    tts_token_buffer.append(token)
                    
                    # Update live translation display
                    current_translation = "".join(translation_buffer)
                    st.session_state.current_translation = current_translation
                    
                    # Check for natural stop (punctuation)
                    should_send_to_tts = False
                    
                    # Check if token contains punctuation (natural stop)
                    if TTS_PUNCTUATION_WAIT and token.strip() in ".,!?;:…":
                        should_send_to_tts = True
                    # Check buffer length
                    elif len("".join(tts_token_buffer)) >= TTS_MAX_BUFFER_LENGTH:
                        should_send_to_tts = True
                    # Check space-based threshold
                    elif token == " " and len("".join(tts_token_buffer).strip()) >= TTS_SPACE_SEND_LENGTH:
                        should_send_to_tts = True
                    
                    if should_send_to_tts:
                        # Send buffered text to TTS (natural stop detected)
                        text_to_speak = "".join(tts_token_buffer)
                        if text_to_speak.strip():
                            await tts_process(text_to_speak)
                        tts_token_buffer = []  # Clear buffer
        
        # Handle any remaining tokens in buffer
        if tts_token_buffer:
            text_to_speak = "".join(tts_token_buffer)
            if text_to_speak.strip():
                await tts_process(text_to_speak)
        
        # Save final translation
        final_translation = "".join(translation_buffer)
        if final_translation:
            # Clean up any multiple-option patterns
            if "/" in final_translation:
                final_translation = final_translation.split("/")[0].strip()
            
            st.session_state.translation_history.append(final_translation)
            st.session_state.llm_messages.append({"role": "assistant", "content": final_translation})
        
    except Exception as e:
        st.error(f"Translation Error: {e}")
        import traceback
        st.code(traceback.format_exc())

async def tts_process(text: str):
    """Process text through ElevenLabs TTS (only called at natural stops)."""
    if not elevenlabs_client or not text.strip():
        return
    
    try:
        voice_id = st.session_state.voice_id
        
        # Generate audio
        audio_chunks = []
        async for audio_chunk in elevenlabs_client.text_to_speech.stream(
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            text=text,
            output_format="pcm_16000",
        ):
            if audio_chunk:
                audio_chunks.append(audio_chunk)
        
        # Combine all chunks and convert to WAV
        if audio_chunks and st.session_state.is_recording:
            audio_data = b"".join(audio_chunks)
            wav_data = pcm_to_wav(audio_data, sample_rate=16000, channels=1)
            st.session_state.audio_outputs.append(wav_data)
            if st.session_state.is_recording:  # Only rerun if still recording
                st.rerun()
        
    except Exception as e:
        st.error(f"TTS Error: {e}")

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert PCM audio data to WAV format for browser playback."""
    wav_buffer = io.BytesIO()
    
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    wav_buffer.seek(0)
    return wav_buffer.read()

# Footer
st.markdown("---")
st.markdown("**StarLife Real-Time Voice Translation** | Built with Streamlit")
