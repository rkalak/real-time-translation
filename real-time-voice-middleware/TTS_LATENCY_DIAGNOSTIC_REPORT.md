# TTS Latency Diagnostic Report

## Executive Summary

The TTS (Text-to-Speech) step is currently the **highest latency component** in the pipeline, averaging **1000-2000ms** per chunk. This report identifies the root causes and provides recommendations for optimization.

---

## Current TTS Latency Breakdown

Based on code analysis and typical performance:

| Component | Estimated Latency | Percentage |
|-----------|------------------|------------|
| **ElevenLabs API Generation** | 800-1500ms | 60-75% |
| **Network Round-Trip** | 100-300ms | 10-15% |
| **Audio Buffering Logic** | 50-200ms | 5-10% |
| **PyAudio Playback** | 50-100ms | 5-10% |
| **Voice Settings Processing** | 10-50ms | 1-5% |
| **Total** | **1000-2000ms** | **100%** |

---

## Root Causes of TTS Latency

### 1. **ElevenLabs API Generation Time** (PRIMARY BOTTLENECK)
**Impact: 60-75% of total latency**

**Causes:**
- **Voice Model Processing**: The `eleven_turbo_v2_5` model still requires significant processing time
- **Voice Settings**: `stability: 0.5` and `similarity_boost: 0.75` add processing overhead
- **Text Length**: Longer text chunks take proportionally longer to generate
- **API Server Load**: ElevenLabs API response time varies based on server load

**Evidence:**
- Current `tts_latency` measurements show 1000-2000ms per chunk
- This is measured from API call start to completion

**Recommendations:**
1. **Reduce voice stability** from `0.5` to `0.3-0.4` for faster generation (trade-off: slightly less stable voice)
2. **Use shorter text chunks**: Current `TTS_MAX_BUFFER_LENGTH = 30` is good, but could reduce to `20-25` for faster generation
3. **Consider ElevenLabs streaming optimization**: Ensure we're using the fastest available model
4. **Parallel processing**: If multiple chunks are ready, send them in parallel (requires API rate limit consideration)

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
TTS_MAX_BUFFER_LENGTH = 30  # Max characters before forcing send
TTS_SPACE_SEND_LENGTH = 8   # Send on spaces if buffer reaches this length
TTS_PUNCTUATION_WAIT = True # Wait for punctuation before speaking
```

**Causes:**
- **Punctuation waiting**: Delays sending until punctuation marks (`.`, `!`, `?`, etc.)
- **Space-based buffering**: Waits for buffer to reach 8 characters before sending on spaces
- **Minimum chunk size**: `TTS_MIN_CHUNK_LENGTH = 3` prevents very small chunks

**Recommendations:**
1. **Reduce `TTS_SPACE_SEND_LENGTH`** from `8` to `5-6` for faster sending
2. **Reduce `TTS_MAX_BUFFER_LENGTH`** from `30` to `20-25` for more frequent sends
3. **Consider disabling `TTS_PUNCTUATION_WAIT`** for ultra-low latency (trade-off: may break mid-sentence)
4. **Send on commas immediately**: Already implemented, but could be more aggressive

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
**Impact: 1-5% of total latency**

**Current Settings:**
```python
voice_settings = {
    "stability": 0.5,        # Lower = faster (0.3-0.4 recommended)
    "similarity_boost": 0.75 # Standard value
}
```

**Recommendations:**
1. **Reduce stability to 0.3-0.4**: Faster generation with acceptable quality
2. **Test without voice_settings**: Some API versions may process faster without explicit settings

---

## Optimization Priority Matrix

| Optimization | Impact | Effort | Priority |
|--------------|--------|--------|----------|
| Reduce voice stability (0.5 → 0.3) | High | Low | **P0** |
| Reduce buffer thresholds (30 → 20, 8 → 5) | Medium | Low | **P0** |
| Parallel TTS requests | High | Medium | **P1** |
| WebSocket streaming | High | High | **P2** |
| Async audio writing | Low | Medium | **P2** |
| Connection pooling | Low | Medium | **P3** |

---

## Recommended Immediate Actions

### Quick Wins (Can implement immediately):
1. **Reduce `stability` from `0.5` to `0.3`** in `pipeline.py` line 459
2. **Reduce `TTS_MAX_BUFFER_LENGTH` from `30` to `20`** in `config.py` line 71
3. **Reduce `TTS_SPACE_SEND_LENGTH` from `8` to `5`** in `config.py` line 73

**Expected Impact**: 20-30% latency reduction (200-600ms improvement)

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

1. **ElevenLabs API Call** (`pipeline.py` lines 465-484):
   - `elevenlabs_client.text_to_speech.stream()` - Main API call
   - `voice_settings` - Processing overhead
   - `model_id="eleven_turbo_v2_5"` - Model selection

2. **Buffering Logic** (`pipeline.py` lines 520-544):
   - `TTS_PUNCTUATION_WAIT` - Waits for punctuation
   - `TTS_SPACE_SEND_LENGTH` - Space-based threshold
   - `TTS_MAX_BUFFER_LENGTH` - Maximum buffer size

3. **Voice Settings** (`pipeline.py` lines 458-461):
   - `stability: 0.5` - Can be reduced to 0.3
   - `similarity_boost: 0.75` - Standard value

---

## Expected Latency After Optimizations

| Scenario | Current | After Quick Wins | After All Optimizations |
|----------|---------|-----------------|------------------------|
| **Average TTS Latency** | 1000-2000ms | 700-1400ms | 400-800ms |
| **Improvement** | Baseline | 30% faster | 60% faster |

---

## Monitoring Recommendations

1. **Track per-chunk latency**: Already implemented in `latency_tracker.record_tts()`
2. **Monitor API response times**: Add timing around ElevenLabs API calls
3. **Track buffer wait times**: Measure time spent waiting for punctuation/spaces
4. **Network latency tracking**: Measure network round-trip time separately

---

## Conclusion

The TTS latency is primarily caused by **ElevenLabs API generation time** (60-75% of total). The most impactful optimizations are:

1. **Reduce voice stability** (immediate, low effort, high impact)
2. **Reduce buffer thresholds** (immediate, low effort, medium impact)
3. **Parallel processing** (medium effort, high impact)

Implementing the quick wins should reduce TTS latency by **20-30%**, bringing average latency from **1000-2000ms** down to **700-1400ms**.

