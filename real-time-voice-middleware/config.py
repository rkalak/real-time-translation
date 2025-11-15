"""
Configuration settings for the real-time translation pipeline.
Adjust these values to control latency, quality, and behavior.
"""

import os

# --- API Keys ---
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# --- LLM System Prompt ---
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

# --- Audio Settings ---
import pyaudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Deepgram prefers 16000 Hz

# --- LATENCY CONFIGURATION ---
# ============================================================================
# HOW TO ADJUST LATENCY:
# ============================================================================
# Adjust these values to control latency at each step (in seconds).
# Lower values = lower latency but may reduce quality/coherence.
# Higher values = higher latency but better quality/coherence.
#
# EXAMPLE ADJUSTMENTS:
# - For ULTRA-LOW latency: Set all delays to 0.0, disable autocorrect/prediction
# - For BALANCED: Use current defaults (good balance of speed and quality)
# - For HIGH QUALITY: Increase delays slightly, enable all features
# ============================================================================

# ASR (Speech-to-Text) latency settings
ASR_BUFFER_DELAY = 0.0  # Delay before processing ASR chunks (0 = immediate, 0.1 = 100ms delay)
ASR_MIN_CHUNK_SIZE = 1  # Minimum words before sending to LLM (1 = send immediately, 3 = wait for 3 words)
ASR_MAX_CHUNK_SIZE = 5  # Maximum words per chunk (split if exceeds this, None = no limit)

# LLM (Translation) latency settings
LLM_AUTOCORRECT_ENABLED = True  # Enable autocorrect for ASR errors (True = better accuracy, False = faster)
LLM_PREDICTION_ENABLED = True  # Enable next-token prediction (True = optimized, False = faster)
LLM_PREDICTION_WINDOW = 3  # Number of previous messages to use for prediction (2-5 recommended)
LLM_TEMPERATURE = 0.3  # Translation creativity (0.1 = very consistent, 0.5 = more varied, 0.7 = creative)
LLM_MAX_RETRIES = 2  # Max retries for autocorrect if first attempt fails (0-3 recommended)
LLM_MODEL = "gpt-4o-mini"  # Model to use: "gpt-4o-mini" (fast, cheap) or "gpt-4o" (slower, better quality)

# TTS (Text-to-Speech) latency settings
TTS_BUFFER_DELAY = 0.05  # Delay before sending to TTS (0 = immediate, 0.1 = 100ms delay for smoother output)
TTS_MAX_BUFFER_LENGTH = 75  # Max characters before forcing send (higher = more coherent, smoother speech)
TTS_MIN_CHUNK_LENGTH = 20  # Minimum characters before sending (higher = smoother, less choppy)
TTS_SPACE_SEND_LENGTH = 15  # Send on spaces if buffer reaches this length (higher = smoother phrases)
TTS_PUNCTUATION_WAIT = True  # Wait for punctuation before speaking (True = coherent, False = faster)

# Latency measurement settings
MEASURE_LATENCY = True  # Enable latency measurement and reporting (True = see metrics, False = no overhead)
LATENCY_REPORT_INTERVAL = 3  # Report average latency every N seconds (5 = every 5 seconds)

