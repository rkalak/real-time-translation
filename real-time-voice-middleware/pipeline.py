import asyncio
import os
import typing
import pyaudio # For microphone input
from openai import AsyncOpenAI # OpenAI client for translation

# These are the correct v5 imports
from deepgram import (
    AsyncDeepgramClient
)
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent
from elevenlabs.client import AsyncElevenLabs

# --- 1. Configuration ---
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Updated prompt for SIMULTANEOUS translation
LLM_SYSTEM_PROMPT = """You are a real-time, simultaneous medical interpreter translating English to Telugu.
You will receive small, incremental chunks of English text.
You must translate these chunks into Telugu IMMEDIATELY as you receive them.

CRITICAL RULES:
1. **SIMULTANEOUS:** Do NOT wait for the full sentence. Start translating the first chunk you get.
2. **INCREMENTAL:** Translate each new chunk and continue the previous translation naturally.
3. **OUTPUT TELUGU ONLY:** Your response must be ONLY Telugu text - no explanations, no meta-commentary.
4. **POLISH:** Remove disfluencies (um, uh, like, you know) and maintain a professional, warm, and empathetic tone.
5. **CONTEXT AWARE:** Use the conversation history to maintain coherence across chunks.
6. **NO META-COMMENTARY:** Do not say "Here is the translation:" or anything similar. Just translate."""

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Deepgram prefers 16000 Hz

def play_pcm_audio(audio_data: bytes, sample_rate: int = RATE, channels: int = CHANNELS) -> None:
    """
    Helper to play raw PCM audio using PyAudio.
    """
    if not audio_data:
        return

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=sample_rate,
            output=True,
        )
        try:
            stream.write(audio_data)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()

# --- 2. Initialize Async Clients ---
# The v5 async client automatically finds the API key from your environment
deepgram_client = AsyncDeepgramClient()

# Configure the OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Configure ElevenLabs client - try explicit key first, then fall back to auto-detection
if ELEVENLABS_API_KEY:
    elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    # Try auto-detection from environment
    elevenlabs_client = AsyncElevenLabs()

# Cache for the voice ID (will be set at startup)
elevenlabs_voice_id: typing.Optional[str] = None

async def get_elevenlabs_voice_id() -> str:
    """
    Get a valid voice ID from ElevenLabs. Tries to find Alisha first,
    then falls back to other preferred voices or the specified default.
    """
    global elevenlabs_voice_id
    
    # If we already have a cached voice ID, return it
    if elevenlabs_voice_id:
        return elevenlabs_voice_id
    
    # Default voice ID (Alisha - Indian female voice for Telugu)
    DEFAULT_VOICE_ID = "ftDdhfYtmfGP0tFlBYA1"
    
    try:
        # Try to get available voices
        voices_response = await elevenlabs_client.voices.get_all()
        
        if voices_response.voices and len(voices_response.voices) > 0:
            # First, try to find Alisha (preferred for Telugu)
            for voice in voices_response.voices:
                if voice.name.lower() == "alisha" or voice.voice_id == DEFAULT_VOICE_ID:
                    elevenlabs_voice_id = voice.voice_id
                    print(f"Using ElevenLabs voice: {voice.name} (ID: {voice.voice_id[:8]}...)")
                    return elevenlabs_voice_id
            
            # If Alisha not found, try other preferred names
            preferred_names = ["Alisha", "Rachel", "Bella", "Elli", "Adam", "Antoni", "Josh", "Arnold", "Sam"]
            
            for voice in voices_response.voices:
                if voice.name in preferred_names:
                    elevenlabs_voice_id = voice.voice_id
                    print(f"Using ElevenLabs voice: {voice.name} (ID: {voice.voice_id[:8]}...)")
                    return elevenlabs_voice_id
            
            # If no preferred voice found, use the first available
            first_voice = voices_response.voices[0]
            elevenlabs_voice_id = first_voice.voice_id
            print(f"Using ElevenLabs voice: {first_voice.name} (ID: {first_voice.voice_id[:8]}...)")
            return elevenlabs_voice_id
        else:
            raise Exception("No voices found in your ElevenLabs account")
            
    except Exception as e:
        print(f"Warning: Could not fetch voices from ElevenLabs: {e}")
        print(f"Using default voice ID: Alisha (ID: {DEFAULT_VOICE_ID[:8]}...)")
        # Fallback to the default voice ID
        elevenlabs_voice_id = DEFAULT_VOICE_ID
        return elevenlabs_voice_id


async def microphone_stream_generator():
    """
    This is our LIVE audio input. It captures audio from the
    microphone and yields it in chunks.
    """
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Microphone stream opened. Start speaking...")

        try:
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    yield data
                    await asyncio.sleep(0.01) # Small sleep to yield control
                except OSError as e:
                    # Handle input overflow - just skip this chunk and continue
                    if e.errno == -9981:  # Input overflowed
                        print("Warning: Audio input overflowed, skipping chunk")
                        await asyncio.sleep(0.01)
                        continue
                    else:
                        raise
        except asyncio.CancelledError:
            print("Microphone stream cancelled.")
    finally:
        print("Microphone stream closing.")
        if stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        try:
            p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")


async def asr_worker(audio_stream, asr_to_llm_queue: asyncio.Queue):
    """
    Worker 1: Implements interim diffing logic.
    Sends ONLY new text chunks to the LLM by subtracting already-sent text.
    """
    print("ASR Worker started. Connecting to Deepgram...")
    # This stores the text we've already sent to the LLM
    last_committed_transcript = ""
    
    try:
        async with deepgram_client.listen.v1.connect(
            model="nova-3-medical",
            language="en-US",
            smart_format="true",
            interim_results="true",
            vad_events="false",
            encoding="linear16",
            sample_rate=str(RATE),
            channels=str(CHANNELS),
        ) as dg_connection:
            
            async def on_message(result):
                nonlocal last_committed_transcript
                
                if isinstance(result, ListenV1ResultsEvent):
                    transcript = result.channel.alternatives[0].transcript
                    if not transcript:
                        return
                    
                    # Filter out system-generated words
                    transcript_lower = transcript.lower().strip()
                    system_words = ["translation", "openai", "asr", "final", "interim", 
                                   "processing", "worker", "tts"]
                    words = transcript_lower.split()
                    if len(words) <= 2:
                        if (transcript_lower.startswith("translation:") or 
                            transcript_lower in system_words or
                            (len(words) == 1 and words[0] in system_words)):
                            return
                    
                    is_final = result.is_final or result.speech_final
                    
                    # INTERIM DIFFING LOGIC: Check if new transcript extends the last one
                    if transcript.startswith(last_committed_transcript):
                        # "Happy Path": Append-only interim (e.g., "how are" -> "how are you")
                        new_text = transcript.removeprefix(last_committed_transcript).strip()
                        
                        if new_text:
                            # Send ONLY the new text chunk
                            print(f"ASR (New Chunk): '{new_text}'")
                            await asr_to_llm_queue.put(new_text)
                            # Update baseline
                            last_committed_transcript = transcript
                    
                    elif not is_final:
                        # "Jitter Path": Prefix changed (e.g., "how are" -> "how old are")
                        # Ignore jittery interim, wait for ASR to stabilize
                        print(f"ASR (Jitter ignored): '{transcript}' vs '{last_committed_transcript}'")
                    
                    if is_final and transcript:
                        # "Final Path": VAD event - send remaining text
                        new_text = transcript.removeprefix(last_committed_transcript).strip()
                        if new_text:
                            print(f"ASR (Final Chunk): '{new_text}'")
                            await asr_to_llm_queue.put(new_text)
                        
                        # Reset baseline for next sentence
                        last_committed_transcript = ""
                        # Send sentence end marker
                        await asr_to_llm_queue.put(None)
            
            dg_connection.on(EventType.MESSAGE, on_message)
            listen_task = asyncio.create_task(dg_connection.start_listening())

            try:
                async for audio_chunk in audio_stream:
                    await dg_connection.send_media(audio_chunk)
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        print(f"Error in ASR worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("ASR Worker finished.")


async def llm_worker(asr_to_llm_queue: asyncio.Queue, llm_to_tts_queue: asyncio.Queue):
    """
    Worker 2: Receives SMALL text chunks, maintains chat history for context,
    and streams translation TOKENS to the TTS worker.
    """
    print("LLM Worker (OpenAI) started. Waiting for text...")
    
    # Maintain conversation history for context across chunks
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    
    while True:
        try:
            # Get the new text chunk from ASR
            new_text_chunk = await asr_to_llm_queue.get()
            
            if new_text_chunk is None:
                # Sentence end marker from ASR - reset history for next sentence
                await llm_to_tts_queue.put(None)
                print("\nLLM: End of sentence. Resetting history.\n")
                messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
                continue

            print(f"OpenAI processing: '{new_text_chunk}'")
            
            # Append the new user chunk to history
            messages.append({"role": "user", "content": new_text_chunk})

            # Get streaming response from OpenAI
            response_stream = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
                temperature=0.3
            )

            # Buffer to hold full translation for this chunk
            translation_buffer = []

            async for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        token = delta.content
                        translation_buffer.append(token)
                        # Stream individual TOKENS to TTS worker
                        await llm_to_tts_queue.put(token)
                        print(token, end="", flush=True)

            print()  # Newline for console readability

            # Add full assistant response to history for context
            final_translation = "".join(translation_buffer)
            if final_translation:
                messages.append({"role": "assistant", "content": final_translation})

        except Exception as e:
            print(f"Error in LLM worker (OpenAI): {e}")
            import traceback
            traceback.print_exc()


async def tts_worker(llm_to_tts_queue: asyncio.Queue):
    """
    Worker 3: Receives TOKENS from LLM, buffers them, and streams audio to speaker.
    Opens PyAudio stream once and plays audio chunks as they arrive.
    """
    print("TTS Worker started. Waiting for text...")
    
    # Initialize PyAudio output stream (opened once, reused)
    p = pyaudio.PyAudio()
    audio_play_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True
    )
    
    # Get voice ID at startup
    voice_id = await get_elevenlabs_voice_id()
    if not voice_id:
        print("FATAL: Could not get ElevenLabs voice ID. Exiting TTS worker.")
        audio_play_stream.stop_stream()
        audio_play_stream.close()
        p.terminate()
        return

    # Configuration constants
    SEND_INTERVAL = 0.3  # Send to ElevenLabs every 300ms if we have text
    MIN_CHUNK_SIZE = 3  # Minimum characters before sending
    
    async def send_text_to_tts(text: str):
        """Send text chunk to ElevenLabs and play audio immediately."""
        if not text:
            return
        
        try:
            async for audio_chunk in elevenlabs_client.text_to_speech.stream(
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                text=text,
                output_format="pcm_16000",
            ):
                if audio_chunk:
                    # Write audio directly to speaker as it arrives
                    audio_play_stream.write(audio_chunk)
        
        except Exception as api_error:
            error_msg = str(api_error)
            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                print(f"ElevenLabs API Key Error: {error_msg}")
            elif "404" in error_msg or "voice_not_found" in error_msg.lower():
                print(f"ElevenLabs Voice Error: {error_msg}")
                global elevenlabs_voice_id
                elevenlabs_voice_id = None
            else:
                print(f"ElevenLabs Error: {error_msg}")
    
    # Buffer to accumulate tokens until we have enough to send
    text_buffer = ""
    last_send_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            try:
                # Get token with timeout to allow periodic sends
                token = await asyncio.wait_for(
                    llm_to_tts_queue.get(), 
                    timeout=SEND_INTERVAL
                )
                
                if token is None:
                    # Sentence end marker - send remaining buffer
                    if text_buffer.strip():
                        await send_text_to_tts(text_buffer.strip())
                        text_buffer = ""
                    continue
                
                # Add token to buffer
                text_buffer += token
                
                # Send if we hit a natural break point (space/punctuation) and have enough text
                if len(text_buffer) >= MIN_CHUNK_SIZE and (
                    token in " .,!?;:" or 
                    asyncio.get_event_loop().time() - last_send_time >= SEND_INTERVAL
                ):
                    if text_buffer.strip():
                        await send_text_to_tts(text_buffer.strip())
                        text_buffer = ""
                        last_send_time = asyncio.get_event_loop().time()
            
            except asyncio.TimeoutError:
                # Timeout - send buffer if we have accumulated text
                if text_buffer.strip() and len(text_buffer.strip()) >= MIN_CHUNK_SIZE:
                    await send_text_to_tts(text_buffer.strip())
                    text_buffer = ""
                    last_send_time = asyncio.get_event_loop().time()
            
    except Exception as e:
        print(f"Error in TTS worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        audio_play_stream.stop_stream()
        audio_play_stream.close()
        p.terminate()
        print("TTS Worker finished.")


async def main():
    """
    Main function to create the queues and start the workers.
    """
    asr_to_llm_queue = asyncio.Queue()
    llm_to_tts_queue = asyncio.Queue()

    # Get the (simulated) LiveKit audio stream
    audio_stream = microphone_stream_generator()

    print("--- Starting Real-Time Translation Pipeline ---")
    try:
        await asyncio.gather(
            asr_worker(audio_stream, asr_to_llm_queue),
            llm_worker(asr_to_llm_queue, llm_to_tts_queue), # Fixed the typo here
            tts_worker(llm_to_tts_queue)
        )
    except KeyboardInterrupt:
        print("\nPipeline shutting down...")

if __name__ == "__main__":
    # Check if API keys are set
    missing_keys = []
    if not DEEPGRAM_API_KEY:
        missing_keys.append("DEEPGRAM_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing_keys.append("ELEVENLABS_API_KEY")
    
    if missing_keys:
        print("Error: One or more API keys are not set.")
        print(f"   Missing: {', '.join(missing_keys)}")
        print("\n   To set them, run:")
        for key in missing_keys:
            print(f"   export {key}=your_api_key_here")
        print("\n   Or add them to your shell profile (.zshrc, .bashrc, etc.)")
        if "ELEVENLABS_API_KEY" in missing_keys:
            print("\n   Get your ElevenLabs API key from: https://elevenlabs.io/app/settings/api-keys")
    else:
        # Validate API key formats (basic checks)
        if ELEVENLABS_API_KEY and len(ELEVENLABS_API_KEY) < 20:
            print("Warning: ELEVENLABS_API_KEY seems too short. Make sure it's the full key.")
        
        print("All API keys are set. Starting pipeline...")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass # Already handled in main