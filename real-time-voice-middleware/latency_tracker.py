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
        self.output_texts = []  # Track all output text sent to TTS
        self.pending_outputs = []  # Track outputs that haven't finished TTS playback yet
        self.completed_outputs = []  # Track outputs that have finished TTS playback
        self.last_tts_finish_time = None  # Track when last TTS finished
        
        # Track latency per output (for showing average per output, not rolling average)
        self.output_latencies = []  # List of dicts: {"llm": time, "tts": time, "output": text}
    
    def record_speech(self):
        """Call this when speech input is received to update last speech time."""
        if MEASURE_LATENCY:
            self.last_speech_time = time.time()
            self.report_scheduled = False  # Cancel any pending report
    
    def _should_report(self):
        """Check if we should report (5 seconds since last TTS finished AND all TTS finished)."""
        if not MEASURE_LATENCY:
            return False
        
        # Only report if:
        # 1. All TTS has finished (no pending outputs)
        # 2. We have completed outputs to show
        # 3. 5 seconds have passed since the last TTS finished (not since input started)
        # 4. We have latency data
        # 5. Report hasn't been scheduled yet
        all_tts_finished = len(self.pending_outputs) == 0
        has_completed_outputs = len(self.completed_outputs) > 0
        has_latency_data = self.asr_times or self.llm_times or self.tts_times
        
        if not (all_tts_finished and has_completed_outputs and has_latency_data):
            return False
        
        # Check if 5 seconds have passed since last TTS finished
        if self.last_tts_finish_time is None:
            return False
        
        current_time = time.time()
        time_since_last_tts = current_time - self.last_tts_finish_time
        
        return (time_since_last_tts >= 5.0 and not self.report_scheduled)
    
    def check_and_report(self):
        """Check if it's time to report and do so if needed. Call this periodically."""
        if self._should_report():
            self.report_scheduled = True
            # Capture completed outputs at this moment (snapshot)
            # This prevents new outputs from being added to this report
            report_outputs = list(self.completed_outputs)
            self.report(report_outputs)
            # Reset after reporting
            self.asr_times = []
            self.llm_times = []
            self.tts_times = []
            self.total_times = []
            self.completed_outputs = []  # Clear completed outputs after reporting
            self.last_tts_finish_time = None  # Reset TTS finish time
            self.output_latencies = []  # Clear per-output latencies after reporting
            # Note: output_texts and pending_outputs are managed separately
            # Flag will be reset when new speech comes in via record_speech()
    
    def record_output(self, text: str):
        """Record output text that was sent to TTS (but not yet finished playing)."""
        if MEASURE_LATENCY:
            if text and text.strip():
                self.output_texts.append(text.strip())
                self.pending_outputs.append(text.strip())
    
    def mark_tts_complete(self, text: str, tts_latency: float = None):
        """Mark that TTS has finished playing for this text."""
        if MEASURE_LATENCY:
            if text and text.strip():
                text_stripped = text.strip()
                # Move from pending to completed
                if text_stripped in self.pending_outputs:
                    self.pending_outputs.remove(text_stripped)
                # Only add to completed if a report hasn't been scheduled yet
                # This prevents outputs from being added after a report is generated
                if not self.report_scheduled and text_stripped not in self.completed_outputs:
                    self.completed_outputs.append(text_stripped)
                    # Track latency for this output (use most recent LLM and TTS times)
                    if tts_latency is not None and self.llm_times:
                        # Get the most recent LLM latency (for this output)
                        recent_llm = self.llm_times[-1] if self.llm_times else 0
                        self.output_latencies.append({
                            "llm": recent_llm,
                            "tts": tts_latency,
                            "total": recent_llm + tts_latency,
                            "output": text_stripped
                        })
                self.last_tts_finish_time = time.time()
    
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
    
    def report(self, report_outputs=None):
        """Print latency statistics.
        
        Args:
            report_outputs: Optional list of outputs to include in the report.
                           If None, uses self.completed_outputs.
        """
        if not MEASURE_LATENCY:
            return
        
        # Use provided outputs or current completed outputs
        if report_outputs is None:
            report_outputs = self.completed_outputs
        
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
        
        # Combine all output texts from the snapshot into one line
        if report_outputs:
            combined_output = " ".join(report_outputs)
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

