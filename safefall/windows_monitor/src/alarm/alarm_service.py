"""
SafeFall Local Acoustic Alarm Service.
Generates repeating beep-beep alarm pattern independent of network connectivity,
with duty-cycle verification mode to prevent microphone acoustic feedback.
"""

import sys
import time
import threading
from typing import Optional

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlarmService:
    def __init__(self, frequency_hz: int = 1500, beep_duration_ms: int = 250, pause_ms: int = 200):
        self.frequency_hz = frequency_hz
        self.beep_duration_ms = beep_duration_ms
        self.pause_ms = pause_ms

        self._is_active = False
        self._verification_mode = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self, verification_mode: bool = False):
        """Starts the repeating acoustic alarm loop in background."""
        with self._lock:
            self._verification_mode = verification_mode
            self._is_paused = False
            if self._is_active:
                return
            self._is_active = True
            self._thread = threading.Thread(target=self._alarm_loop, daemon=True)
            self._thread.start()

    def set_verification_mode(self, enabled: bool):
        """
        In verification mode, plays intermittent reminder beeps with wide silence intervals
        so the microphone can capture voice input cleanly without speaker acoustic feedback.
        """
        with self._lock:
            self._verification_mode = enabled

    def pause(self):
        """Temporarily pauses beeping (e.g. while speech synthesis prompt is speaking)."""
        with self._lock:
            self._is_paused = True

    def resume(self):
        """Resumes beeping after speech synthesis finishes."""
        with self._lock:
            self._is_paused = False

    def stop(self):
        """Stops the repeating alarm immediately."""
        with self._lock:
            self._is_active = False
            self._verification_mode = False
            self._is_paused = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)

    def is_ringing(self) -> bool:
        with self._lock:
            return self._is_active and not self._is_paused

    def play_warning_chime(self):
        """Plays 2 quick distinct beeps synchronously or in a helper thread."""
        def _chime():
            self._play_tone(self.frequency_hz, 180)
            time.sleep(0.12)
            self._play_tone(self.frequency_hz, 180)
        threading.Thread(target=_chime, daemon=True).start()

    def _alarm_loop(self):
        while True:
            with self._lock:
                if not self._is_active:
                    break
                is_paused = self._is_paused
                is_verif = self._verification_mode

            if is_paused:
                time.sleep(0.1)
                continue

            if is_verif:
                # Duty-cycle mode for voice response:
                # 1 short beep (120ms), followed by a 3.5s silent window for microphone capture
                self._play_tone(self.frequency_hz, 120)
                # Sleep in small increments so stop() responds immediately
                for _ in range(35):
                    with self._lock:
                        if not self._is_active or not self._verification_mode or self._is_paused:
                            break
                    time.sleep(0.1)
            else:
                # Full emergency alarm mode: Continuous loud double-beep pattern
                self._play_tone(self.frequency_hz, self.beep_duration_ms)
                time.sleep(self.pause_ms / 1000.0)

                with self._lock:
                    if not self._is_active or self._verification_mode or self._is_paused:
                        continue

                self._play_tone(self.frequency_hz, self.beep_duration_ms)
                time.sleep(self.pause_ms * 3 / 1000.0)

    def _play_tone(self, freq: int, duration_ms: int):
        if HAS_WINSOUND:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass
        else:
            time.sleep(duration_ms / 1000.0)
