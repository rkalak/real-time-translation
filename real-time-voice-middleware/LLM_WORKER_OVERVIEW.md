# LLM Worker (Text-to-Text Translation) Overview

## Purpose
The `llm_worker` function translates English ASR chunks to Spanish in real-time, with error correction and optimization features.

## Processing Flow

### 1. Input Reception
- Receives text chunks from `asr_to_llm_queue` (from ASR worker)
- Handles sentence end markers (`None`) by resetting conversation history

### 2. Autocorrect Step (Optional, adds latency)
**Function**: `autocorrect_text()`
- **Purpose**: Fixes ASR errors using conversation history
- **When**: Runs BEFORE translation for each chunk
- **Latency Impact**: ~200-500ms per chunk (additional API call)
- **Config**: `LLM_AUTOCORRECT_ENABLED` (default: True)
- **Process**:
  1. Builds prompt with recent conversation history (last N messages)
  2. Sends to OpenAI API with autocorrect instructions
  3. Returns corrected English text
- **Example**: "how r u" → "how are you", "I am" → "I'm"

### 3. Next-Token Prediction (Optional, non-blocking)
**Function**: `predict_next_tokens()`
- **Purpose**: Predicts likely next words for optimization (background task)
- **When**: Runs asynchronously (doesn't block translation)
- **Latency Impact**: None (runs in background)
- **Config**: `LLM_PREDICTION_ENABLED` (default: True)
- **Process**:
  1. Uses conversation history to predict next 1-3 words
  2. Runs as `asyncio.create_task()` (non-blocking)
  3. Results not currently used (optimization for future)

### 4. Translation Step (Main processing)
**Function**: OpenAI Chat Completions API
- **Purpose**: Translates English to Spanish
- **Model**: `gpt-4o-mini` (configurable via `LLM_MODEL`)
- **Temperature**: 0.3 (configurable via `LLM_TEMPERATURE`)
- **Streaming**: Yes (tokens streamed in real-time)
- **Latency**: ~500-1000ms per chunk (varies by chunk size)
- **Process**:
  1. Appends corrected text to conversation history
  2. Sends to OpenAI with system prompt + full history
  3. Streams tokens as they're generated
  4. Filters out multiple-option patterns (e.g., "el/la" → "el")
  5. Sends tokens immediately to TTS worker

### 5. Token Filtering
- **Multiple-option detection**: Filters out patterns like "el/la/los/las"
- **Process**: Takes text before "/", skips until space/punctuation
- **Purpose**: Ensures clean, single-option translations

### 6. History Management
- Maintains full conversation history for context
- Resets on sentence end (`None` marker)
- Adds both user inputs and assistant translations to history

## Latency Breakdown

### Per Chunk Processing Time:
1. **Autocorrect**: ~200-500ms (if enabled)
2. **Translation**: ~500-1000ms (main step)
3. **Token streaming**: Real-time (no additional latency)
4. **Total**: ~700-1500ms per chunk

### Current Latency Tracking:
- `record_llm()` tracks only translation time (not autocorrect)
- Shows rolling average of all translation times
- **Issue**: Doesn't show per-output average (fixed in new version)

## Configuration Options

| Setting | Default | Impact |
|---------|---------|--------|
| `LLM_AUTOCORRECT_ENABLED` | True | Adds ~200-500ms per chunk, improves accuracy |
| `LLM_PREDICTION_ENABLED` | True | No latency impact (background), future optimization |
| `LLM_TEMPERATURE` | 0.3 | Lower = faster, more consistent |
| `LLM_MODEL` | gpt-4o-mini | gpt-4o-mini = faster/cheaper, gpt-4o = slower/better |

## Optimization Opportunities

### 1. Disable Autocorrect (Fastest)
- **Impact**: Saves ~200-500ms per chunk
- **Trade-off**: May have more ASR errors in translation
- **When to use**: If ASR quality is already good

### 2. Use GPT-3.5-Turbo (Faster model)
- **Impact**: ~50% faster than gpt-4o-mini
- **Trade-off**: Slightly lower translation quality
- **When to use**: If speed is critical and quality is acceptable

### 3. Lower Temperature
- **Impact**: ~10-20% faster token generation
- **Trade-off**: Less variation in translations
- **Current**: 0.3 (already optimized)

### 4. Batch Multiple Chunks
- **Impact**: Reduces API call overhead
- **Trade-off**: Increases latency (waits for batch)
- **Not recommended**: Defeats purpose of real-time translation

### 5. Remove Next-Token Prediction
- **Impact**: No latency impact (already non-blocking)
- **Trade-off**: None (not currently used)
- **Recommendation**: Can disable if not planning to use

## Current Issues

1. **Autocorrect adds significant latency** (~200-500ms per chunk)
2. **Rolling average doesn't show per-output latency** (fixed)
3. **No caching of common phrases** (could speed up repeated phrases)
4. **Full history sent with each request** (increases token count)

## Recommendations

1. **For lowest latency**: Disable autocorrect (`LLM_AUTOCORRECT_ENABLED = False`)
2. **For best quality**: Keep autocorrect enabled, use gpt-4o model
3. **For balanced**: Current settings (autocorrect + gpt-4o-mini)
4. **Monitor per-output latency**: Now tracked and displayed in reports

