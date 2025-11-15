"""
Latency tracking and measurement utilities.
"""

import time
from config import MEASURE_LATENCY, LATENCY_REPORT_INTERVAL


class LatencyTracker:
    """Tracks latency at each step of the pipeline."""
    def __init__(self):
        self.asr_times = []
        self.llm_times = []
        self.tts_times = []
        self.total_times = []
        self.last_speech_time = time.time()  # Track when speech was last received
        self.report_scheduled = False
        self.output_texts = []  # Track all output text for the report
    
    def record_speech(self):
        """Call this when speech input is received to update last speech time."""
        if MEASURE_LATENCY:
            self.last_speech_time = time.time()
            self.report_scheduled = False  # Cancel any pending report
    
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
            # Reset after reporting
            self.asr_times = []
            self.llm_times = []
            self.tts_times = []
            self.total_times = []
            self.output_texts = []
            # Flag will be reset when new speech comes in via record_speech()
    
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
        
        # Combine all output texts into one line
        if self.output_texts:
            combined_output = " ".join(self.output_texts)
            print(f"\nCOMBINED OUTPUT:")
            print(f"{combined_output}")
        
        print("="*60 + "\n")
        
        # Keep only recent samples (last 100)
        self.asr_times = self.asr_times[-100:]
        self.llm_times = self.llm_times[-100:]
        self.tts_times = self.tts_times[-100:]
        self.total_times = self.total_times[-100:]


# Global latency tracker instance
latency_tracker = LatencyTracker()

