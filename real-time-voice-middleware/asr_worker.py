"""
ASR (Speech-to-Text) Worker using Deepgram.
Handles microphone input and converts speech to text.
"""

import asyncio
import time
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent

from config import (
    DEEPGRAM_API_KEY, ASR_BUFFER_DELAY, ASR_MIN_CHUNK_SIZE, ASR_MAX_CHUNK_SIZE,
    RATE, CHANNELS, MEASURE_LATENCY
)
from latency_tracker import latency_tracker

# Initialize Deepgram client
deepgram_client = AsyncDeepgramClient()


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
                asr_start = time.time()
                
                if isinstance(result, ListenV1ResultsEvent):
                    transcript = result.channel.alternatives[0].transcript
                    if not transcript:
                        return
                    
                    # Apply buffer delay if configured
                    if ASR_BUFFER_DELAY > 0:
                        await asyncio.sleep(ASR_BUFFER_DELAY)
                    
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
                            # Check minimum chunk size
                            word_count = len(new_text.split())
                            if word_count >= ASR_MIN_CHUNK_SIZE:
                                # Check maximum chunk size - split if too long
                                if ASR_MAX_CHUNK_SIZE and word_count > ASR_MAX_CHUNK_SIZE:
                                    # Split into smaller chunks
                                    words = new_text.split()
                                    for i in range(0, len(words), ASR_MAX_CHUNK_SIZE):
                                        chunk = " ".join(words[i:i + ASR_MAX_CHUNK_SIZE])
                                        print(f"[INPUT]  {chunk}")
                                        await asr_to_llm_queue.put(chunk)
                                else:
                                    # Send the complete chunk
                                    print(f"[INPUT]  {new_text}")
                                    await asr_to_llm_queue.put(new_text)
                                
                                # Update baseline
                                last_committed_transcript = transcript
                                
                                # Record ASR latency and mark speech received
                                asr_latency = time.time() - asr_start
                                latency_tracker.record_asr(asr_latency)
                                latency_tracker.record_speech()
                    
                    elif not is_final:
                        # "Jitter Path": Prefix changed (e.g., "how are" -> "how old are")
                        # Ignore jittery interim, wait for ASR to stabilize
                        print(f"ASR (Jitter ignored): '{transcript}' vs '{last_committed_transcript}'")
                    
                    if is_final and transcript:
                        # "Final Path": VAD event - send remaining text
                        new_text = transcript.removeprefix(last_committed_transcript).strip()
                        if new_text:
                            # Check maximum chunk size - split if too long
                            word_count = len(new_text.split())
                            if ASR_MAX_CHUNK_SIZE and word_count > ASR_MAX_CHUNK_SIZE:
                                # Split into smaller chunks
                                words = new_text.split()
                                for i in range(0, len(words), ASR_MAX_CHUNK_SIZE):
                                    chunk = " ".join(words[i:i + ASR_MAX_CHUNK_SIZE])
                                    print(f"[INPUT]  {chunk}")
                                    await asr_to_llm_queue.put(chunk)
                            else:
                                print(f"[INPUT]  {new_text}")
                                await asr_to_llm_queue.put(new_text)
                            
                            # Record ASR latency and mark speech received
                            asr_latency = time.time() - asr_start
                            latency_tracker.record_asr(asr_latency)
                            latency_tracker.record_speech()
                        
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

