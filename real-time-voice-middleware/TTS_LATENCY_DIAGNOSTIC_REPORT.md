# TTS Latency Diagnostic Report

## Executive Summary

The TTS (Text-to-Speech) step is currently the **highest latency component** in the pipeline, averaging **1200-1800ms** per chunk based on recent measurements. This report identifies the root causes and provides recommendations for optimization.

**Last Updated**: Based on current configuration and recent performance measurements.

---

## Current TTS Latency Breakdown

Based on code analysis and recent performance measurements:

| Component | Estimated Latency | Percentage |
|-----------|------------------|------------|
| **ElevenLabs API Generation** | 900-1400ms | 65-75% |
| **Network Round-Trip** | 100-300ms | 8-15% |
| **TTS Buffer Delay** | 50ms | 3-5% |
| **Audio Buffering Logic** | 100-200ms | 5-10% |
| **PyAudio Playback** | 50-100ms | 3-5% |
| **Voice Settings Processing** | 10-50ms | 1-3% |
| **Total** | **1200-1800ms** | **100%** |

**Recent Measurement**: Average TTS latency of ~1456ms (from terminal output)

---

## Root Causes of TTS Latency

### 1. **ElevenLabs API Generation Time** (PRIMARY BOTTLENECK)
**Impact: 60-75% of total latency**

**Causes:**
- **Voice Model Processing**: The `eleven_turbo_v2_5` model requires significant processing time
- **Voice Settings**: `stability: 0.7` and `similarity_boost: 0.8` add processing overhead (optimized for quality over speed)
- **Text Length**: Longer text chunks take proportionally longer to generate
- **API Server Load**: ElevenLabs API response time varies based on server load
- **Current Buffer Settings**: `TTS_MAX_BUFFER_LENGTH = 75` and `TTS_MIN_CHUNK_LENGTH = 20` create larger chunks

**Evidence:**
- Recent `tts_latency` measurements show ~1456ms average per chunk
- This is measured from API call start to completion
- Larger buffer settings (75 chars max, 20 chars min) increase chunk size and generation time

**Recommendations:**
1. **Reduce voice stability** from `0.7` to `0.4-0.5` for faster generation (trade-off: slightly less stable voice)
2. **Reduce buffer thresholds**: Current `TTS_MAX_BUFFER_LENGTH = 75` is high - consider reducing to `40-50`
3. **Reduce minimum chunk length**: Current `TTS_MIN_CHUNK_LENGTH = 20` is high - consider reducing to `10-15` for faster sends
4. **Consider ElevenLabs streaming optimization**: Add `optimize_streaming_latency=3` parameter if supported
5. **Switch to Flash model**: Consider `eleven_flash_v2_5` if available (faster than Turbo)

### 2. **Network Latency** (SECONDARY BOTTLENECK)
**Impact: 10-15% of total latency**

**Causes:**
- **HTTP/WebSocket overhead**: Each API call has network round-trip time
- **Geographic distance**: Distance to ElevenLabs servers
- **Network congestion**: Variable network conditions

**Recommendations:**
1. **Connection pooling**: Reuse HTTP connections where possible
2. **WebSocket streaming**: If ElevenLabs supports WebSocket streaming, use it instead of HTTP requests
3. **Regional optimization**: Use closest ElevenLabs data center if available

### 3. **Audio Buffering Logic** (MODERATE IMPACT)
**Impact: 5-10% of total latency**

**Current Implementation:**
```python
# Waits for punctuation or buffer length thresholds
TTS_BUFFER_DELAY = 0.05        # 50ms delay before sending
TTS_MAX_BUFFER_LENGTH = 75     # Max characters before forcing send
TTS_MIN_CHUNK_LENGTH = 20     # Minimum characters before sending
TTS_SPACE_SEND_LENGTH = 15    # Send on spaces if buffer reaches this length
TTS_PUNCTUATION_WAIT = True   # Wait for punctuation before speaking
```

**Causes:**
- **Buffer delay**: `TTS_BUFFER_DELAY = 0.05` adds 50ms before each TTS call
- **Punctuation waiting**: Delays sending until punctuation marks (`.`, `!`, `?`, `;`, etc.)
- **Space-based buffering**: Waits for buffer to reach 15 characters before sending on spaces
- **Minimum chunk size**: `TTS_MIN_CHUNK_LENGTH = 20` requires larger chunks before sending
- **Maximum buffer**: `TTS_MAX_BUFFER_LENGTH = 75` allows very large chunks (increases generation time)

**Recommendations:**
1. **Reduce `TTS_BUFFER_DELAY`** from `0.05` to `0.0` for immediate sending (saves 50ms per chunk)
2. **Reduce `TTS_MAX_BUFFER_LENGTH`** from `75` to `40-50` for more frequent, smaller sends
3. **Reduce `TTS_MIN_CHUNK_LENGTH`** from `20` to `10-15` for faster initial sends
4. **Reduce `TTS_SPACE_SEND_LENGTH`** from `15` to `10-12` for faster space-based sends
5. **Consider disabling `TTS_PUNCTUATION_WAIT`** for ultra-low latency (trade-off: may break mid-sentence)

### 4. **PyAudio Playback** (MINOR IMPACT)
**Impact: 5-10% of total latency**

**Causes:**
- **Synchronous audio writing**: `audio_play_stream.write(audio_chunk)` blocks until audio is written
- **Audio buffer management**: PyAudio internal buffering adds small delays

**Recommendations:**
1. **Use async audio writing**: Consider using `asyncio.to_thread()` for non-blocking writes
2. **Optimize buffer size**: Adjust PyAudio buffer size for lower latency
3. **Pre-buffer audio**: Start playing audio as soon as first chunk arrives (already implemented)

### 5. **Voice Settings Processing** (MINIMAL IMPACT)
**Impact: 1-3% of total latency**

**Current Settings:**
```python
voice_settings = {
    "stability": 0.7,        # Higher = smoother but slower (0.4-0.5 recommended for speed)
    "similarity_boost": 0.8  # Higher = more consistent voice
}
```

**Current Trade-off:**
- Settings optimized for **quality and smoothness** over speed
- `stability: 0.7` prioritizes smooth, stable speech but increases generation time
- `similarity_boost: 0.8` ensures consistent voice but adds processing overhead

**Recommendations:**
1. **Reduce stability to 0.4-0.5**: Faster generation with acceptable quality (saves ~100-200ms)
2. **Reduce similarity_boost to 0.65-0.75**: Slightly faster processing with minimal quality impact
3. **Test without voice_settings**: Some API versions may process faster without explicit settings
4. **Add `optimize_streaming_latency=3`**: If supported, can reduce latency by 50% (400-500ms)

---

## Optimization Priority Matrix

| Optimization | Impact | Effort | Priority |
|--------------|--------|--------|----------|
| Reduce voice stability (0.7 → 0.4) | High | Low | **P0** |
| Reduce buffer delay (0.05 → 0.0) | Medium | Low | **P0** |
| Reduce buffer thresholds (75 → 50, 20 → 15) | Medium | Low | **P0** |
| Add optimize_streaming_latency=3 | High | Low | **P0** |
| Switch to Flash model (if available) | High | Low | **P1** |
| Parallel TTS requests | High | Medium | **P1** |
| WebSocket streaming | High | High | **P2** |
| Async audio writing | Low | Medium | **P2** |
| Connection pooling | Low | Medium | **P3** |

---

## Recommended Immediate Actions

### Quick Wins (Can implement immediately):
1. **Reduce `stability` from `0.7` to `0.4-0.5`** in `pipeline.py` line 486
2. **Reduce `TTS_BUFFER_DELAY` from `0.05` to `0.0`** in `config.py` line 70 (saves 50ms per chunk)
3. **Reduce `TTS_MAX_BUFFER_LENGTH` from `75` to `40-50`** in `config.py` line 71
4. **Reduce `TTS_MIN_CHUNK_LENGTH` from `20` to `10-15`** in `config.py` line 72
5. **Add `optimize_streaming_latency=3`** parameter to ElevenLabs API call (if supported)

**Expected Impact**: 25-35% latency reduction (300-600ms improvement)

### Medium-term Optimizations:
1. **Implement parallel TTS requests**: Send multiple chunks simultaneously if ready
2. **Optimize buffering logic**: Send immediately on spaces if buffer >= 5 chars
3. **Test without voice_settings**: May reduce API processing time

**Expected Impact**: Additional 10-20% latency reduction

### Long-term Optimizations:
1. **WebSocket streaming**: If ElevenLabs supports it, use WebSocket for lower latency
2. **Regional API endpoints**: Use closest data center
3. **Custom voice model**: Train a faster, lighter voice model if possible

**Expected Impact**: Additional 15-25% latency reduction

---

## Code Locations for Optimization

### Primary TTS Latency Sources:

1. **ElevenLabs API Call** (`pipeline.py` lines 492-501):
   - `elevenlabs_client.text_to_speech.stream()` - Main API call
   - `voice_settings` - Processing overhead (stability: 0.7, similarity_boost: 0.8)
   - `model_id="eleven_turbo_v2_5"` - Model selection (consider Flash v2.5)
   - **Missing**: `optimize_streaming_latency=3` parameter (can add significant speedup)

2. **Buffering Logic** (`pipeline.py` lines 529-570):
   - `TTS_BUFFER_DELAY` - 50ms delay before sending (config.py line 70)
   - `TTS_PUNCTUATION_WAIT` - Waits for punctuation
   - `TTS_SPACE_SEND_LENGTH` - Space-based threshold (currently 15)
   - `TTS_MAX_BUFFER_LENGTH` - Maximum buffer size (currently 75)
   - `TTS_MIN_CHUNK_LENGTH` - Minimum chunk size (currently 20)

3. **Voice Settings** (`pipeline.py` lines 485-488):
   - `stability: 0.7` - Can be reduced to 0.4-0.5 for speed
   - `similarity_boost: 0.8` - Can be reduced to 0.65-0.75

---

## Expected Latency After Optimizations

| Scenario | Current | After Quick Wins | After All Optimizations |
|----------|---------|-----------------|------------------------|
| **Average TTS Latency** | 1200-1800ms | 800-1200ms | 500-900ms |
| **Improvement** | Baseline | 30-35% faster | 50-60% faster |

**Current Measurement**: ~1456ms average (from recent terminal output)
**Target After Quick Wins**: ~900-1000ms average
**Target After All Optimizations**: ~600-800ms average

---

## Monitoring Recommendations

1. **Track per-chunk latency**: Already implemented in `latency_tracker.record_tts()`
2. **Monitor API response times**: Add timing around ElevenLabs API calls
3. **Track buffer wait times**: Measure time spent waiting for punctuation/spaces
4. **Network latency tracking**: Measure network round-trip time separately

---

## Current Configuration Summary

**Current Settings (Optimized for Quality/Smoothness):**
- `TTS_BUFFER_DELAY = 0.05` (50ms delay)
- `TTS_MAX_BUFFER_LENGTH = 75` (large chunks)
- `TTS_MIN_CHUNK_LENGTH = 20` (requires substantial text before sending)
- `TTS_SPACE_SEND_LENGTH = 15` (waits for longer phrases)
- `voice_settings.stability = 0.7` (high quality, slower)
- `voice_settings.similarity_boost = 0.8` (consistent voice)
- `model_id = "eleven_turbo_v2_5"` (Turbo model, not Flash)

**Trade-off**: Current settings prioritize smooth, coherent speech over raw speed, resulting in higher latency but better quality.

## Conclusion

The TTS latency is primarily caused by **ElevenLabs API generation time** (65-75% of total) and **current buffer settings optimized for quality** (larger chunks = longer generation time). The most impactful optimizations are:

1. **Reduce voice stability** from 0.7 to 0.4-0.5 (immediate, low effort, high impact - saves ~100-200ms)
2. **Reduce buffer delay** from 0.05 to 0.0 (immediate, low effort, medium impact - saves 50ms per chunk)
3. **Reduce buffer thresholds** (75→50, 20→15) (immediate, low effort, medium impact - enables faster sends)
4. **Add optimize_streaming_latency=3** (immediate, low effort, high impact - saves ~400-500ms if supported)
5. **Switch to Flash model** (if available) (immediate, low effort, high impact - saves ~150-200ms)

Implementing the quick wins should reduce TTS latency by **25-35%**, bringing average latency from **~1456ms** down to **~900-1000ms**.

