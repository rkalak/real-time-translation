"""
CONSOLIDATED REAL-TIME VOICE TRANSLATION PIPELINE
==================================================
This is a single-file version of the entire pipeline for easy copying.
All code from config.py, latency_tracker.py, asr_worker.py, and pipeline.py
has been consolidated into this file.

DO NOT USE THIS FILE FOR PRODUCTION - Use the modular structure instead.
This file is for copying/pasting code elsewhere.
"""

import asyncio
import os
import typing
import time
import pyaudio
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# LLM System Prompt
LLM_SYSTEM_PROMPT = """You are a real-time, simultaneous medical interpreter translating English to Spanish.
You will receive small, incremental chunks of English text.
You must translate these chunks into Spanish IMMEDIATELY as you receive them.

CRITICAL RULES:
1. **SIMULTANEOUS:** Do NOT wait for the full sentence. Start translating the first chunk you get.
2. **INCREMENTAL:** Translate each new chunk and continue the previous translation naturally.
3. **OUTPUT SPANISH ONLY:** Your response must be ONLY Spanish text - no explanations, no meta-commentary.
4. **CHOOSE ONE TRANSLATION:** When multiple translations are possible (e.g., "the" could be "el/la/los/las"), 
   choose the MOST LIKELY option based on context. NEVER output multiple options separated by slashes.
   If uncertain, choose the most common/default option (e.g., "el" for "the" when gender is unclear).
5. **EXACT TRANSLATION:** Translate EXACTLY what is in the input. Do NOT add extra words, phrases, or context.
   Remove disfluencies (um, uh, like, you know) but do NOT add explanatory words, emotional context, or 
   additional information that was not in the original English text.
6. **TONE:** Maintain a professional, warm, and empathetic tone through word choice and phrasing, but WITHOUT
   adding extra words. Adjust tone through translation choices, not by adding content.
7. **CONTEXT AWARE:** Use the conversation history to maintain coherence across chunks, but do NOT infer or add
   information that wasn't explicitly stated in the input.
8. **NO META-COMMENTARY:** Do not say "Here is the translation:" or anything similar. Just translate.
9. **TRANSLATE ALL CHUNKS:** Every input chunk must be translated. Do not skip or combine chunks."""

# Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Deepgram prefers 16000 Hz

# ASR (Speech-to-Text) latency settings
ASR_BUFFER_DELAY = 0.0
ASR_MIN_CHUNK_SIZE = 3
ASR_MAX_CHUNK_SIZE = 5

# LLM (Translation) latency settings
LLM_AUTOCORRECT_ENABLED = True
LLM_PREDICTION_ENABLED = True
LLM_PREDICTION_WINDOW = 3
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 2
LLM_MODEL = "gpt-4o-mini"

# TTS (Text-to-Speech) latency settings
TTS_BUFFER_DELAY = 0.0
TTS_MAX_BUFFER_LENGTH = 30
TTS_MIN_CHUNK_LENGTH = 3
TTS_SPACE_SEND_LENGTH = 8
TTS_PUNCTUATION_WAIT = True

# Latency measurement settings
MEASURE_LATENCY = True
LATENCY_REPORT_INTERVAL = 5

# ============================================================================
# LATENCY TRACKER
# ============================================================================

class LatencyTracker:
    """Tracks latency at each step of the pipeline."""
    def __init__(self):
        self.asr_times = []
        self.llm_times = []
        self.tts_times = []
        self.total_times = []
        self.last_speech_time = time.time()
        self.report_scheduled = False
        self.output_texts = []
    
    def record_speech(self):
        """Call this when speech input is received to update last speech time."""
        if MEASURE_LATENCY:
            self.last_speech_time = time.time()
            self.report_scheduled = False
    
    def _should_report(self):
        """Check if we should report (5 seconds since last speech)."""
        if not MEASURE_LATENCY:
            return False
        current_time = time.time()
        time_since_speech = current_time - self.last_speech_time
        return time_since_speech >= 5.0 and (self.asr_times or self.llm_times or self.tts_times) and not self.report_scheduled
    
    def check_and_report(self):
        """Check if it's time to report and do so if needed. Call this periodically."""
        if self._should_report():
            self.report_scheduled = True
            self.report()
            self.asr_times = []
            self.llm_times = []
            self.tts_times = []
            self.total_times = []
            self.output_texts = []
    
    def record_output(self, text: str):
        """Record output text that was sent to TTS."""
        if MEASURE_LATENCY:
            if text and text.strip():
                self.output_texts.append(text.strip())
    
    def record_asr(self, duration: float):
        """Record ASR processing time."""
        if MEASURE_LATENCY:
            self.asr_times.append(duration)
    
    def record_llm(self, duration: float):
        """Record LLM processing time."""
        if MEASURE_LATENCY:
            self.llm_times.append(duration)
    
    def record_tts(self, duration: float):
        """Record TTS processing time."""
        if MEASURE_LATENCY:
            self.tts_times.append(duration)
    
    def record_total(self, duration: float):
        """Record total end-to-end time."""
        if MEASURE_LATENCY:
            self.total_times.append(duration)
    
    def report(self):
        """Print latency statistics."""
        if not MEASURE_LATENCY:
            return
        
        print("\n" + "="*60)
        print("LATENCY REPORT")
        print("="*60)
        
        if self.asr_times:
            avg_asr = sum(self.asr_times) / len(self.asr_times)
            print(f"ASR (Speech-to-Text): {avg_asr*1000:.2f}ms avg (last {len(self.asr_times)} samples)")
        
        if self.llm_times:
            avg_llm = sum(self.llm_times) / len(self.llm_times)
            print(f"LLM (Translation): {avg_llm*1000:.2f}ms avg (last {len(self.llm_times)} samples)")
        
        if self.tts_times:
            avg_tts = sum(self.tts_times) / len(self.tts_times)
            print(f"TTS (Text-to-Speech): {avg_tts*1000:.2f}ms avg (last {len(self.tts_times)} samples)")
        
        if self.total_times:
            avg_total = sum(self.total_times) / len(self.total_times)
            print(f"TOTAL (End-to-End): {avg_total*1000:.2f}ms avg (last {len(self.total_times)} samples)")
        
        if self.output_texts:
            combined_output = " ".join(self.output_texts)
            print(f"\nCOMBINED OUTPUT:")
            print(f"{combined_output}")
        
        print("="*60 + "\n")
        
        self.asr_times = self.asr_times[-100:]
        self.llm_times = self.llm_times[-100:]
        self.tts_times = self.tts_times[-100:]
        self.total_times = self.total_times[-100:]

latency_tracker = LatencyTracker()

# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def play_pcm_audio(audio_data: bytes, sample_rate: int = RATE, channels: int = CHANNELS) -> None:
    """Helper to play raw PCM audio using PyAudio."""
    if not audio_data:
        return
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=channels, rate=sample_rate, output=True)
        try:
            stream.write(audio_data)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

deepgram_client = AsyncDeepgramClient()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

if ELEVENLABS_API_KEY:
    elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    elevenlabs_client = AsyncElevenLabs()

elevenlabs_voice_id: typing.Optional[str] = None

async def get_elevenlabs_voice_id() -> str:
    """Get a valid voice ID from ElevenLabs. Prioritizes female voices for Spanish translation."""
    global elevenlabs_voice_id
    if elevenlabs_voice_id:
        return elevenlabs_voice_id
    try:
        voices_response = await elevenlabs_client.voices.get_all()
        if voices_response.voices and len(voices_response.voices) > 0:
            preferred_female_names = [
                "Alisha", "Sarah", "Laura", "Alice", "Matilda", "Jessica", 
                "Lily", "Rachel", "Bella", "Elli", "Charlotte", "Emily", "Nova"
            ]
            for voice in voices_response.voices:
                if voice.name in preferred_female_names:
                    elevenlabs_voice_id = voice.voice_id
                    print(f"Using ElevenLabs voice: {voice.name} (ID: {voice.voice_id[:8]}...)")
                    return elevenlabs_voice_id
            female_keywords = ["sarah", "laura", "alice", "matilda", "jessica", "lily", 
                             "rachel", "bella", "charlotte", "emily", "nova", "alisha"]
            for voice in voices_response.voices:
                if any(keyword in voice.name.lower() for keyword in female_keywords):
                    elevenlabs_voice_id = voice.voice_id
                    print(f"Using ElevenLabs voice: {voice.name} (ID: {voice.voice_id[:8]}...)")
                    return elevenlabs_voice_id
            first_voice = voices_response.voices[0]
            elevenlabs_voice_id = first_voice.voice_id
            print(f"Using ElevenLabs voice: {first_voice.name} (ID: {first_voice.voice_id[:8]}...)")
            return elevenlabs_voice_id
        else:
            raise Exception("No voices found in your ElevenLabs account")
    except Exception as e:
        print(f"Warning: Could not fetch voices from ElevenLabs: {e}")
        fallback_voice_id = "EXAVITQu4vr4xnSDxMaL"  # Sarah
        print(f"Using fallback voice ID: Sarah (ID: {fallback_voice_id[:8]}...)")
        elevenlabs_voice_id = fallback_voice_id
        return elevenlabs_voice_id

async def microphone_stream_generator():
    """Live audio input from microphone."""
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Microphone stream opened. Start speaking...")
        try:
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    yield data
                    await asyncio.sleep(0.01)
                except OSError as e:
                    if e.errno == -9981:
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

# ============================================================================
# ASR WORKER
# ============================================================================

async def asr_worker(audio_stream, asr_to_llm_queue: asyncio.Queue):
    """Worker 1: Implements interim diffing logic. Sends ONLY new text chunks to the LLM."""
    print("ASR Worker started. Connecting to Deepgram...")
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
                asr_start = time.time()
                if isinstance(result, ListenV1ResultsEvent):
                    transcript = result.channel.alternatives[0].transcript
                    if not transcript:
                        return
                    if ASR_BUFFER_DELAY > 0:
                        await asyncio.sleep(ASR_BUFFER_DELAY)
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
                    if transcript.startswith(last_committed_transcript):
                        new_text = transcript.removeprefix(last_committed_transcript).strip()
                        if new_text:
                            word_count = len(new_text.split())
                            if word_count >= ASR_MIN_CHUNK_SIZE:
                                if ASR_MAX_CHUNK_SIZE and word_count > ASR_MAX_CHUNK_SIZE:
                                    words = new_text.split()
                                    for i in range(0, len(words), ASR_MAX_CHUNK_SIZE):
                                        chunk = " ".join(words[i:i + ASR_MAX_CHUNK_SIZE])
                                        print(f"[INPUT]  {chunk}")
                                        await asr_to_llm_queue.put(chunk)
                                else:
                                    print(f"[INPUT]  {new_text}")
                                    await asr_to_llm_queue.put(new_text)
                                last_committed_transcript = transcript
                                asr_latency = time.time() - asr_start
                                latency_tracker.record_asr(asr_latency)
                                latency_tracker.record_speech()
                    elif not is_final:
                        print(f"ASR (Jitter ignored): '{transcript}' vs '{last_committed_transcript}'")
                    if is_final and transcript:
                        new_text = transcript.removeprefix(last_committed_transcript).strip()
                        if new_text:
                            word_count = len(new_text.split())
                            if ASR_MAX_CHUNK_SIZE and word_count > ASR_MAX_CHUNK_SIZE:
                                words = new_text.split()
                                for i in range(0, len(words), ASR_MAX_CHUNK_SIZE):
                                    chunk = " ".join(words[i:i + ASR_MAX_CHUNK_SIZE])
                                    print(f"[INPUT]  {chunk}")
                                    await asr_to_llm_queue.put(chunk)
                            else:
                                print(f"[INPUT]  {new_text}")
                                await asr_to_llm_queue.put(new_text)
                            asr_latency = time.time() - asr_start
                            latency_tracker.record_asr(asr_latency)
                            latency_tracker.record_speech()
                        last_committed_transcript = ""
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

# ============================================================================
# LLM WORKER
# ============================================================================

async def llm_worker(asr_to_llm_queue: asyncio.Queue, llm_to_tts_queue: asyncio.Queue):
    """Worker 2: Enhanced LLM worker with autocorrect, next-token prediction, and latency tracking."""
    print("LLM Worker (OpenAI gpt-4o-mini) started. Waiting for text...")
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    prediction_context = []
    
    async def autocorrect_text(text: str, conversation_history: list) -> str:
        """Use LLM to autocorrect ASR errors using conversation history."""
        if not LLM_AUTOCORRECT_ENABLED:
            return text
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
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an autocorrect assistant for English speech-to-text. You ONLY correct ASR errors in English. You NEVER translate."},
                    {"role": "user", "content": autocorrect_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            corrected = response.choices[0].message.content.strip()
            corrected = corrected.strip('"').strip("'")
            if corrected and corrected != text:
                print(f"[AUTOCORRECT]  '{text}' -> '{corrected}'")
            return corrected if corrected else text
        except Exception as e:
            print(f"Autocorrect error: {e}, using original text")
            return text
    
    async def predict_next_tokens(conversation_history: list) -> str:
        """Predict likely next tokens based on conversation history."""
        if not LLM_PREDICTION_ENABLED or len(conversation_history) < 2:
            return ""
        recent_history = conversation_history[-LLM_PREDICTION_WINDOW:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        prediction_prompt = f"""Based on this conversation, predict what the user might say next (1-3 words).
This is for optimization only - do NOT translate.

Conversation:
{history_text}

Predict next 1-3 words the user might say:"""
        try:
            response = await openai_client.chat.completions.create(
                model=LLM_MODEL,
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
            start_time = time.time()
            new_text_chunk = await asr_to_llm_queue.get()
            if new_text_chunk is None:
                await llm_to_tts_queue.put(None)
                print("\nLLM: End of sentence. Resetting history.\n")
                messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
                prediction_context = []
                continue
            corrected_chunk = await autocorrect_text(new_text_chunk, messages)
            if LLM_PREDICTION_ENABLED:
                asyncio.create_task(predict_next_tokens(messages))
            print(f"LLM processing: '{corrected_chunk}'")
            messages.append({"role": "user", "content": corrected_chunk})
            translation_start = time.time()
            response_stream = await openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                temperature=LLM_TEMPERATURE
            )
            translation_buffer = []
            skip_until_space = False
            print(f"[TRANSLATION]  ", end="", flush=True)
            async for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        token = delta.content
                        translation_buffer.append(token)
                        if "/" in token:
                            parts = token.split("/", 1)
                            if parts[0]:
                                await llm_to_tts_queue.put(parts[0])
                                print(parts[0], end="", flush=True)
                            skip_until_space = True
                            continue
                        if skip_until_space:
                            if token.strip() in " .,!?;:" or token.isspace():
                                skip_until_space = False
                                await llm_to_tts_queue.put(token)
                                print(token, end="", flush=True)
                            continue
                        await llm_to_tts_queue.put(token)
                        print(token, end="", flush=True)
            translation_end = time.time()
            llm_latency = translation_end - translation_start
            latency_tracker.record_llm(llm_latency)
            print()
            final_translation = "".join(translation_buffer)
            if "/" in final_translation:
                cleaned = final_translation.split("/")[0].strip()
                if cleaned and cleaned != final_translation:
                    print(f"Warning: Cleaned translation from '{final_translation}' to '{cleaned}'")
                    final_translation = cleaned
            if final_translation:
                messages.append({"role": "assistant", "content": final_translation})
        except Exception as e:
            print(f"Error in LLM worker (OpenAI): {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# TTS WORKER
# ============================================================================

async def tts_worker(llm_to_tts_queue: asyncio.Queue):
    """Worker 3: Receives TOKENS, buffers them into coherent sentence fragments, and streams audio."""
    print("TTS Worker (Coherence Mode) started. Waiting for text...")
    p = pyaudio.PyAudio()
    audio_play_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    voice_id = await get_elevenlabs_voice_id()
    if not voice_id:
        print("FATAL: Could not get ElevenLabs voice ID. Exiting TTS worker.")
        audio_play_stream.stop_stream()
        audio_play_stream.close()
        p.terminate()
        return
    
    async def latency_checker():
        while True:
            await asyncio.sleep(1)
            latency_tracker.check_and_report()
    
    latency_check_task = asyncio.create_task(latency_checker())
    
    async def send_text_to_tts(text: str):
        """Send a coherent text chunk to ElevenLabs and play audio immediately."""
        if not text.strip():
            return
        if TTS_BUFFER_DELAY > 0:
            await asyncio.sleep(TTS_BUFFER_DELAY)
        tts_start = time.time()
        print(f"\n[OUTPUT]  {text}")
        latency_tracker.record_output(text)
        try:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
            }
            try:
                async for audio_chunk in elevenlabs_client.text_to_speech.stream(
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5",
                    text=text,
                    output_format="pcm_16000",
                    voice_settings=voice_settings,
                ):
                    if audio_chunk:
                        audio_play_stream.write(audio_chunk)
            except TypeError:
                async for audio_chunk in elevenlabs_client.text_to_speech.stream(
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5",
                    text=text,
                    output_format="pcm_16000",
                ):
                    if audio_chunk:
                        audio_play_stream.write(audio_chunk)
            tts_latency = time.time() - tts_start
            latency_tracker.record_tts(tts_latency)
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
    
    text_buffer = ""
    try:
        while True:
            token = await llm_to_tts_queue.get()
            if token is None:
                if text_buffer.strip():
                    await send_text_to_tts(text_buffer.strip())
                text_buffer = ""
                continue
            text_buffer += token
            should_send = False
            buffer_length = len(text_buffer.strip())
            if TTS_PUNCTUATION_WAIT and token.strip() in ".,!?;:…":
                should_send = True
            elif not TTS_PUNCTUATION_WAIT and token.strip() == ",":
                should_send = True
            elif token == " " and buffer_length >= TTS_SPACE_SEND_LENGTH:
                should_send = True
            elif len(text_buffer) >= TTS_MAX_BUFFER_LENGTH:
                should_send = True
            elif not TTS_PUNCTUATION_WAIT and buffer_length >= TTS_MIN_CHUNK_LENGTH:
                should_send = True
            if should_send:
                await send_text_to_tts(text_buffer)
                text_buffer = ""
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

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main function to create the queues and start the workers."""
    asr_to_llm_queue = asyncio.Queue()
    llm_to_tts_queue = asyncio.Queue()
    audio_stream = microphone_stream_generator()
    print("--- Starting Real-Time Translation Pipeline ---")
    try:
        await asyncio.gather(
            asr_worker(audio_stream, asr_to_llm_queue),
            llm_worker(asr_to_llm_queue, llm_to_tts_queue),
            tts_worker(llm_to_tts_queue)
        )
    except KeyboardInterrupt:
        print("\nPipeline shutting down...")

if __name__ == "__main__":
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
        if ELEVENLABS_API_KEY and len(ELEVENLABS_API_KEY) < 20:
            print("Warning: ELEVENLABS_API_KEY seems too short. Make sure it's the full key.")
        print("All API keys are set. Starting pipeline...")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass

