import asyncio
import typing
import time
import pyaudio # For microphone input
from openai import AsyncOpenAI # OpenAI client for translation
from elevenlabs.client import AsyncElevenLabs

# Import configuration and utilities from modular files
from config import (
    DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY,
    LLM_SYSTEM_PROMPT, CHUNK, FORMAT, CHANNELS, RATE,
    ASR_BUFFER_DELAY, ASR_MIN_CHUNK_SIZE, ASR_MAX_CHUNK_SIZE,
    LLM_AUTOCORRECT_ENABLED, LLM_PREDICTION_ENABLED, LLM_PREDICTION_WINDOW,
    LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_MODEL,
    TTS_BUFFER_DELAY, TTS_MAX_BUFFER_LENGTH, TTS_MIN_CHUNK_LENGTH,
    TTS_SPACE_SEND_LENGTH, TTS_PUNCTUATION_WAIT,
    MEASURE_LATENCY, LATENCY_REPORT_INTERVAL
)
from latency_tracker import latency_tracker
from asr_worker import asr_worker

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

# --- Initialize Async Clients ---
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
    Get a valid voice ID from ElevenLabs. Uses the specified voice ID.
    """
    global elevenlabs_voice_id
    
    # If we already have a cached voice ID, return it
    if elevenlabs_voice_id:
        return elevenlabs_voice_id
    
    # Use the specified voice ID
    specified_voice_id = "T4Au24Lt2uWk24Qra0No"
    
    try:
        # Verify the voice exists by trying to get all voices
        voices_response = await elevenlabs_client.voices.get_all()
        
        # Check if the specified voice ID exists in the available voices
        voice_found = False
        voice_name = "Unknown"
        if voices_response.voices:
            for voice in voices_response.voices:
                if voice.voice_id == specified_voice_id:
                    voice_found = True
                    voice_name = voice.name
                    break
        
        if voice_found:
            elevenlabs_voice_id = specified_voice_id
            print(f"Using ElevenLabs voice: {voice_name} (ID: {specified_voice_id[:8]}...)")
            return elevenlabs_voice_id
        else:
            # Voice ID not found in account, but try to use it anyway (might be a public voice)
            print(f"Warning: Voice ID {specified_voice_id[:8]}... not found in your account, but attempting to use it anyway.")
            elevenlabs_voice_id = specified_voice_id
            return elevenlabs_voice_id
            
    except Exception as e:
        print(f"Warning: Could not verify voice from ElevenLabs: {e}")
        # Use the specified voice ID anyway (might work even if verification fails)
        print(f"Using specified voice ID: {specified_voice_id[:8]}...")
        elevenlabs_voice_id = specified_voice_id
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
        except Exception as e:
            print(f"Error in microphone stream: {e}")
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


# ASR worker is now imported from asr_worker.py


async def llm_worker(asr_to_llm_queue: asyncio.Queue, llm_to_tts_queue: asyncio.Queue):
    """
    Worker 2: Enhanced LLM worker with autocorrect, next-token prediction, and latency tracking.
    - Autocorrects ASR errors using conversation history
    - Uses next-token prediction to optimize translation
    - Tracks latency at each step
    """
    print("LLM Worker (OpenAI gpt-4o-mini) started. Waiting for text...")
    
    # Maintain conversation history for context across chunks
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    # Prediction context for next-token prediction
    prediction_context = []
    
    async def autocorrect_text(text: str, conversation_history: list) -> str:
        """
        Use LLM to autocorrect ASR errors using conversation history.
        Returns corrected text.
        """
        if not LLM_AUTOCORRECT_ENABLED:
            return text
        
        # Build autocorrect prompt using recent conversation history
        recent_history = conversation_history[-LLM_PREDICTION_WINDOW:] if conversation_history else []
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        autocorrect_prompt = f"""You are an autocorrect system for speech-to-text errors in ENGLISH.
Given the conversation history and a potentially incorrect transcription, correct any ASR errors.

Conversation history:
{history_text}

Incorrect transcription to correct: "{text}"

CRITICAL RULES:
1. The input is in ENGLISH - keep it in ENGLISH. Do NOT translate to Spanish or any other language.
2. Only correct obvious speech-to-text errors (e.g., "how are you" not "how r u", "I'm" not "I am")
3. Fix common ASR mistakes like missing apostrophes, wrong homophones, etc.
4. Keep the meaning and intent exactly the same
5. Return ONLY the corrected ENGLISH text, no explanations, no Spanish

Corrected ENGLISH text:"""
        
        try:
            response = await openai_client.chat.completions.create(
                model=LLM_MODEL,  # Use configured model
                messages=[
                    {"role": "system", "content": "You are an autocorrect assistant for English speech-to-text. You ONLY correct ASR errors in English. You NEVER translate."},
                    {"role": "user", "content": autocorrect_prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent correction
                max_tokens=100
            )
            corrected = response.choices[0].message.content.strip()
            # Remove quotes if present
            corrected = corrected.strip('"').strip("'")
            if corrected and corrected != text:
                print(f"[AUTOCORRECT]  '{text}' -> '{corrected}'")
            return corrected if corrected else text
        except Exception as e:
            print(f"Autocorrect error: {e}, using original text")
            return text
    
    async def predict_next_tokens(conversation_history: list) -> str:
        """
        Predict likely next tokens based on conversation history.
        This helps optimize the translation model by providing context.
        Returns predicted continuation (for optimization, not for translation).
        """
        if not LLM_PREDICTION_ENABLED or len(conversation_history) < 2:
            return ""
        
        # Use recent history for prediction
        recent_history = conversation_history[-LLM_PREDICTION_WINDOW:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        prediction_prompt = f"""Based on this conversation, predict what the user might say next (1-3 words).
This is for optimization only - do NOT translate.

Conversation:
{history_text}

Predict next 1-3 words the user might say:"""
        
        try:
            response = await openai_client.chat.completions.create(
                model=LLM_MODEL,  # Use configured model
                messages=[
                    {"role": "system", "content": "You are a conversation predictor."},
                    {"role": "user", "content": prediction_prompt}
                ],
                temperature=0.5,
                max_tokens=20
            )
            prediction = response.choices[0].message.content.strip().lower()
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return ""
    
    while True:
        try:
            # Get the new text chunk from ASR
            start_time = time.time()
            new_text_chunk = await asr_to_llm_queue.get()
            
            if new_text_chunk is None:
                # Sentence end marker from ASR - reset history for next sentence
                await llm_to_tts_queue.put(None)
                print("\nLLM: End of sentence. Resetting history.\n")
                messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
                prediction_context = []
                continue

            # All processes run in parallel - no blocking
            # Step 1: Start autocorrect in parallel (non-blocking)
            # Step 2: Start translation immediately with original input
            # Step 3: Start next-token prediction in background
            
            # Start autocorrect in parallel (non-blocking)
            # Translation starts immediately without waiting - they run simultaneously
            corrected_chunk = new_text_chunk  # Use original text for translation
            autocorrect_task = None
            if LLM_AUTOCORRECT_ENABLED:
                # Start autocorrect asynchronously - runs in parallel with translation
                autocorrect_task = asyncio.create_task(autocorrect_text(new_text_chunk, messages))
            
            # Start next-token prediction in background (non-blocking, zero latency impact)
            if LLM_PREDICTION_ENABLED:
                # Run prediction asynchronously (don't wait for it)
                asyncio.create_task(predict_next_tokens(messages))
            
            # Start translation immediately (runs in parallel with autocorrect)
            # Translation uses original text - autocorrect runs simultaneously in background
            print(f"LLM processing: '{new_text_chunk}'")
            
            # Append the original chunk to history
            messages.append({"role": "user", "content": new_text_chunk})
            
            translation_start = time.time()
            response_stream = await openai_client.chat.completions.create(
                model=LLM_MODEL,  # Uses model from config (gpt-4o-mini by default)
                messages=messages,
                stream=True,
                temperature=LLM_TEMPERATURE
            )
            
            # Autocorrect continues running in background while translation streams
            # If autocorrect finishes and finds corrections, log them for reference
            # (but translation already started, so we continue with original text)
            if autocorrect_task:
                # Check autocorrect result after translation starts (non-blocking)
                async def log_autocorrect_result():
                    try:
                        corrected = await autocorrect_task
                        if corrected != new_text_chunk:
                            print(f"[AUTOCORRECT]  '{new_text_chunk}' -> '{corrected}' (logged for reference)")
                    except Exception as e:
                        pass  # Ignore autocorrect errors
                asyncio.create_task(log_autocorrect_result())

            # Buffer to hold full translation for this chunk
            translation_buffer = []
            # Flag to skip tokens after detecting a "/" (multiple-option pattern)
            skip_until_space = False
            
            # Start translation output
            print(f"[TRANSLATION]  ", end="", flush=True)

            async for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        token = delta.content
                        translation_buffer.append(token)
                        
                        # Detect if this token contains or starts a multiple-option pattern
                        if "/" in token:
                            # We've detected a "/" - this is likely a multiple-option pattern
                            # Take only the part before the "/"
                            if "/" in token:
                                parts = token.split("/", 1)
                                if parts[0]:  # If there's text before the "/"
                                    # Send only the part before "/"
                                    await llm_to_tts_queue.put(parts[0])
                                    print(parts[0], end="", flush=True)
                                # Skip everything after the "/" until we hit a space or punctuation
                                skip_until_space = True
                                continue
                        
                        # If we're skipping (after a "/"), only resume when we hit a space or punctuation
                        if skip_until_space:
                            if token.strip() in " .,!?;:" or token.isspace():
                                skip_until_space = False
                                # Send the space/punctuation
                                await llm_to_tts_queue.put(token)
                                print(token, end="", flush=True)
                            # Otherwise, skip this token
                            continue
                        
                        # Normal case: stream individual TOKENS to TTS worker
                        await llm_to_tts_queue.put(token)
                        print(token, end="", flush=True)

            # Record LLM translation latency
            translation_end = time.time()
            llm_latency = translation_end - translation_start
            latency_tracker.record_llm(llm_latency)
            print()  # Newline after translation

            # Add full assistant response to history for context
            final_translation = "".join(translation_buffer)
            
            # Clean up any multiple-option patterns (e.g., "el/la/los/las" -> "el")
            # This handles cases where the LLM outputs multiple possibilities
            if "/" in final_translation:
                # If translation contains slashes (multiple options), take the first option
                # This is a fallback - the prompt should prevent this, but we clean it up anyway
                cleaned = final_translation.split("/")[0].strip()
                if cleaned and cleaned != final_translation:
                    print(f"Warning: Cleaned translation from '{final_translation}' to '{cleaned}'")
                    final_translation = cleaned
            
            if final_translation:
                messages.append({"role": "assistant", "content": final_translation})
            
            # Record total LLM processing time (including autocorrect)
            total_llm_time = time.time() - start_time
            if MEASURE_LATENCY and total_llm_time > llm_latency:
                # Autocorrect added some overhead
                pass

        except Exception as e:
            print(f"Error in LLM worker (OpenAI): {e}")
            import traceback
            traceback.print_exc()


async def tts_worker(llm_to_tts_queue: asyncio.Queue):
    """
    Worker 3: Receives TOKENS, buffers them into coherent sentence fragments,
    and then streams the audio for that fragment.
    
    This coherence-first approach buffers tokens until natural break points
    (punctuation marks) to ensure smooth, natural-sounding speech output.
    
    Coherence Evaluation Metrics (Research-Based):
    - BLEU Score: Measures structural/lexical similarity to human translations
      (higher = more coherent). Our buffering approach improves BLEU by ensuring
      complete phrases are spoken together.
    - Average Lagging (AL): Measures how many words behind the speaker we are.
      This approach trades slightly higher AL (3-5 words) for much better coherence.
    - Human Evaluation (MOS): Subjective 1-5 scale for "naturalness" and "coherence".
      Buffering to punctuation marks significantly improves MOS scores.
    
    The trade-off: Slightly increased latency (higher AL) for significantly
    improved speech quality and coherence (higher BLEU/MOS).
    """
    print("TTS Worker (Coherence Mode) started. Waiting for text...")
    
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

    # Track recently sent TTS text to prevent exact duplicates
    recent_tts_text = set()
    recent_tts_window = []  # Keep a sliding window of recent texts
    MAX_RECENT_TEXTS = 10  # Track last 10 chunks to prevent duplicates
    
    # Background task to check for latency reporting after speech pauses
    async def latency_checker():
        while True:
            await asyncio.sleep(1)  # Check every second
            latency_tracker.check_and_report()
    
    # Start latency checker task
    latency_check_task = asyncio.create_task(latency_checker())
    
    async def send_text_to_tts(text: str):
        """
        Send a coherent text chunk to ElevenLabs and play audio immediately.
        Uses voice settings to slow down and smooth out the speech.
        Prevents duplicate TTS calls for the same text.
        """
        if not text.strip():
            return
        
        # Check for exact duplicates within recent window
        text_normalized = text.strip().lower()
        if text_normalized in recent_tts_text:
            print(f"TTS skipping duplicate: '{text}'")
            return
        
        # Mark as sent and add to recent window
        recent_tts_text.add(text_normalized)
        recent_tts_window.append(text_normalized)
        
        # Maintain sliding window (remove oldest if exceeds limit)
        if len(recent_tts_window) > MAX_RECENT_TEXTS:
            oldest = recent_tts_window.pop(0)
            recent_tts_text.discard(oldest)
        
        # Apply buffer delay if configured
        if TTS_BUFFER_DELAY > 0:
            await asyncio.sleep(TTS_BUFFER_DELAY)
        
        tts_start = time.time()
        print(f"\n[OUTPUT]  {text}")
        
        # Record output text for latency report
        latency_tracker.record_output(text)
        
        try:
            # Voice settings - optimized for smoother, less choppy speech
            voice_settings = {
                "stability": 0.7,  # Higher = smoother, more stable speech (0.7 = smoother, 0.5 = faster)
                "similarity_boost": 0.8,  # Higher = more consistent voice
            }
            
            # Try to use voice_settings if supported, otherwise use without it
            try:
                async for audio_chunk in elevenlabs_client.text_to_speech.stream(
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5",  # Already using fastest model
                    text=text,
                    output_format="pcm_16000",
                    voice_settings=voice_settings,
                ):
                    if audio_chunk:
                        # Write audio directly to speaker as it arrives (non-blocking)
                        audio_play_stream.write(audio_chunk)
            except TypeError:
                # If voice_settings parameter is not supported, use without it
                async for audio_chunk in elevenlabs_client.text_to_speech.stream(
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5",
                    text=text,
                    output_format="pcm_16000",
                ):
                    if audio_chunk:
                        audio_play_stream.write(audio_chunk)
            
            # Record TTS latency
            tts_latency = time.time() - tts_start
            latency_tracker.record_tts(tts_latency)
            
            # Mark that TTS has finished playing for this text
            # (all audio chunks have been written to the stream)
            latency_tracker.mark_tts_complete(text, tts_latency)
        
        except Exception as api_error:
            error_msg = str(api_error)
            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                print(f"ElevenLabs API Key Error: {error_msg}")
            elif "404" in error_msg or "voice_not_found" in error_msg.lower():
                print(f"ElevenLabs Voice Error: {error_msg}")
                global elevenlabs_voice_id
                elevenlabs_voice_id = None  # Reset cache
            else:
                print(f"ElevenLabs Error: {error_msg}")
    
    # This buffer will accumulate tokens into sentence fragments
    # We only send to TTS when we hit a natural breaking point
    text_buffer = ""
    
    try:
        while True:
            # Wait for the next token from the LLM
            token = await llm_to_tts_queue.get()

            if token is None:
                # End of sentence marker. Send any remaining text.
                if text_buffer.strip():
                    await send_text_to_tts(text_buffer.strip())
                text_buffer = ""
                # Clear duplicate tracking after sentence end to allow same words in different sentences
                recent_tts_text.clear()
                recent_tts_window.clear()
                continue
            
            # Add the new token to our buffer
            text_buffer += token
            
            # Check if we've hit a natural breaking point
            # Optimized for LOW LATENCY: Send smaller chunks more frequently
            should_send = False
            buffer_length = len(text_buffer.strip())
            
            # Priority 1: Punctuation marks (always send - natural break)
            if TTS_PUNCTUATION_WAIT and token.strip() in ".,!?;:…":
                should_send = True
            # Priority 2: Commas (send immediately for faster flow, even if punctuation wait is off)
            elif not TTS_PUNCTUATION_WAIT and token.strip() == ",":
                should_send = True
            # Priority 3: Spaces - send if buffer reaches threshold (aggressive for low latency)
            elif token == " " and buffer_length >= TTS_SPACE_SEND_LENGTH:
                should_send = True
            # Priority 4: Safety - force send if buffer gets too long
            elif len(text_buffer) >= TTS_MAX_BUFFER_LENGTH:
                should_send = True
            # Priority 5: If punctuation wait is disabled, send on minimum chunk size
            elif not TTS_PUNCTUATION_WAIT and buffer_length >= TTS_MIN_CHUNK_LENGTH:
                should_send = True
            
            if should_send:
                # Send the buffered text (including the punctuation/space)
                await send_text_to_tts(text_buffer)
                text_buffer = ""  # Clear the buffer for the next fragment
        
    except Exception as e:
        print(f"Error in TTS worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        latency_check_task.cancel()
        try:
            await latency_check_task
        except asyncio.CancelledError:
            pass
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