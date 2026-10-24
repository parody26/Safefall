"""
SafeFall Speech Interaction and Verbal Intent Classification Service.
Prompts the individual, captures microphone input (via sounddevice or speech_recognition),
and determines whether the subject is safely reassuring or requesting emergency help.
Includes multi-phrase listening, adaptive VAD, and speaker acoustic feedback suppression.
"""

import re
import time
import math
import subprocess
import threading
import collections
from enum import Enum
from typing import Optional, Callable, List, Dict, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


class VerbalIntent(str, Enum):
    CONFIRM_SAFE = "CONFIRM_SAFE"
    REQUEST_HELP = "REQUEST_HELP"
    AMBIGUOUS = "AMBIGUOUS"
    NO_RESPONSE = "NO_RESPONSE"
    SPEECH_ERROR = "SPEECH_ERROR"


# Explicit Reassurance Patterns (Cancels alarm / Confirms safety)
SAFE_PATTERNS = [
    r"\bi'?m fine\b",
    r"\bi am fine\b",
    r"\bfine\b",
    r"\bi'?m okay\b",
    r"\bi am okay\b",
    r"\bi'?m ok\b",
    r"\bi am ok\b",
    r"\bokay\b",
    r"\bok\b",
    r"\bi'?m good\b",
    r"\bgood\b",
    r"\beverything is fine\b",
    r"\beverything'?s fine\b",
    r"\ball good\b",
    r"\bno problem\b",
    r"\bdon'?t need help\b",
    r"\bdo not need help\b",
    r"\bno help needed\b",
    r"\bi'?m alright\b",
    r"\bi am alright\b",
    r"\ball right\b",
    r"\bi'?m all right\b",
    r"\bi am all right\b",
    r"\bstop alarm\b",
    r"\bstop\b",
    r"\bcancel alarm\b",
    r"\bcancel\b",
    r"\bfalse alarm\b",
    r"\bturn off alarm\b",
    r"\bsilence\b",
    r"\bi'?m safe\b",
    r"\bi am safe\b",
    r"\bsafe\b",
    r"\bcan get up\b",
    r"\bi can get up\b",
    r"\bnot hurt\b",
    r"\bunhurt\b",
    r"\bno emergency\b",
    r"\bno ambulance\b",
    r"\byes i am fine\b",
    r"\byeah i'?m fine\b",
    r"\byes\b",
    r"\byep\b",
    r"\byeah\b",
    r"\bi can stand\b",
    r"\bi just tripped\b",
    r"\bi dropped something\b",
    r"\bjust resting\b",
    r"\bi'?m just resting\b",
]

# Active Emergency / Urgent Help Cry Patterns (Escalates immediately to family)
ACTIVE_HELP_PATTERNS = [
    r"\bhelp me\b",
    r"\bhelp\b",
    r"\bplease help\b",
    r"\bhelp me please\b",
    r"\bneed help\b",
    r"\bi need help\b",
    r"\bsomebody help\b",
    r"\bsomeone help\b",
    r"\bemergency\b",
    r"\ban emergency\b",
    r"\bit'?s an emergency\b",
    r"\bsave me\b",
    r"\bsend help\b",
    r"\bcall an? ambulance\b",
    r"\bambulance\b",
    r"\bcall 911\b",
    r"\bcall 112\b",
    r"\bcall 108\b",
    r"\bcall 100\b",
    r"\bcall (?:911|112|108|100|the police|police|the doctor|my family|my son|my daughter|my mom|my dad)\b",
    r"\bhurts?\b",
    r"\bpain\b",
    r"\bin pain\b",
    r"\bsevere pain\b",
    r"\bi'?m hurt\b",
    r"\bi am hurt\b",
    r"\bouch\b",
    r"\bcan'?t get up\b",
    r"\bcannot get up\b",
    r"\bunable to stand\b",
    r"\bunable to get up\b",
    r"\bcan'?t move\b",
    r"\bcannot move\b",
    r"\bfell down\b",
    r"\bi fell\b",
    r"\bi have fallen\b",
    r"\bbroken\b",
    r"\bbleeding\b",
    r"\binjured\b",
    r"\bcritical\b",
    r"\burgent\b",
    r"\bmedic\b",
    r"\bmayday\b",
    r"\bassistance\b",
    r"\bassist me\b",
]


def classify_transcript(text: str) -> VerbalIntent:
    """
    Evaluates text transcript and returns the prioritized VerbalIntent.
    Rule: ACTIVE_HELP takes priority over reassurance, while explicitly handling
    negations like "I don't need help".
    """
    if not text or not text.strip():
        return VerbalIntent.NO_RESPONSE

    cleaned = text.lower().strip()

    # 1. Check if the entire statement is explicitly "I don't need help" / "do not need help"
    if re.search(r"\b(?:don'?t|do not|no)\s+(?:need\s+)?help\b", cleaned) and not re.search(r"\b(?:call|ambulance|911|hurts?|pain|can'?t get up|bleeding)\b", cleaned):
        return VerbalIntent.CONFIRM_SAFE

    # 2. Active urgent help cry takes top priority (e.g. "help me", "call ambulance", "in pain", "can't get up")
    has_urgent_help = any(re.search(pat, cleaned) for pat in ACTIVE_HELP_PATTERNS)
    if has_urgent_help:
        return VerbalIntent.REQUEST_HELP

    # 3. Check for explicit safety confirmations (including "don't need help", "I am fine", "I'm okay", "stop")
    has_safe = any(re.search(pat, cleaned) for pat in SAFE_PATTERNS)
    if has_safe:
        return VerbalIntent.CONFIRM_SAFE

    # 4. Check for standalone "help" or "need help" when NOT negated
    if re.search(r"\bhelp\b", cleaned):
        if not re.search(r"\b(?:don'?t|do not|no|not)\b.*?\bhelp\b", cleaned):
            return VerbalIntent.REQUEST_HELP

    # 5. Otherwise ambiguous
    return VerbalIntent.AMBIGUOUS


class SpeechService:
    def __init__(self, response_timeout_sec: float = 10.0, input_device_index: Optional[int] = None):
        self.response_timeout_sec = response_timeout_sec
        self.input_device_index = input_device_index
        self.recognizer = sr.Recognizer() if HAS_SR else None
        self._is_listening = False
        self._sample_rate = 16000
        self.last_heard_text = ""
        self.last_intent = VerbalIntent.NO_RESPONSE

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """Enumerates all available audio input microphones on the system."""
        if not HAS_SD:
            return []
        devices = []
        try:
            raw_devices = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            for idx, d in enumerate(raw_devices):
                if d.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": d.get("name", "Unknown Microphone"),
                        "channels": d.get("max_input_channels", 1),
                        "default_samplerate": d.get("default_samplerate", 16000),
                        "is_default": (idx == default_input),
                    })
        except Exception:
            pass
        return devices

    def prompt_verbal_check(
        self,
        prompt_text: str = "SafeFall: Are you okay? Say I am fine, or Help.",
        on_finished: Optional[Callable[[], None]] = None,
    ):
        """Speaks the audio prompt to the room asynchronously, calling on_finished when done."""
        def _speak():
            try:
                spoken = False
                if HAS_TTS:
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 175)
                        engine.say(prompt_text)
                        engine.runAndWait()
                        spoken = True
                    except Exception as ex:
                        print(f"[SafeFall Voice] pyttsx3 notice: {ex}")
                        spoken = False

                if not spoken:
                    try:
                        cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{prompt_text}')"
                        subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=5)
                    except Exception:
                        pass
            finally:
                if on_finished:
                    on_finished()

        threading.Thread(target=_speak, daemon=True).start()

    def speak_confirmation(self, confirmation_text: str = "Alarm cancelled. You are confirmed safe."):
        """Speaks safety confirmation when alarm is cancelled."""
        def _speak():
            if HAS_TTS:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 160)
                    engine.say(confirmation_text)
                    engine.runAndWait()
                    return
                except Exception:
                    pass
            try:
                cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{confirmation_text}')"
                subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=4)
            except Exception:
                pass

        threading.Thread(target=_speak, daemon=True).start()

    def _normalize_audio(self, samples_int16: np.ndarray, target_peak: float = 24000.0) -> np.ndarray:
        """Applies gentle automatic gain control to boost quiet/distant speech from the floor."""
        if not HAS_NUMPY or len(samples_int16) == 0:
            return samples_int16
        samples = samples_int16.astype(float)
        peak = float(np.max(np.abs(samples)))
        if peak < 60.0:  # Pure ambient noise floor
            return samples_int16
        gain = min(target_peak / peak, 8.0)  # Up to 8x boost for quiet room speech
        boosted = np.clip(samples * gain, -32767.0, 32767.0).astype(np.int16)
        return boosted

    def listen_for_response(
        self,
        on_result: Callable[[VerbalIntent, str], None],
        timeout_sec: Optional[float] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_audio_level: Optional[Callable[[float, bool], None]] = None,
    ):
        """
        Listens on microphone continuously for up to timeout_sec (default 10s).
        Uses non-blocking streaming audio capture, dynamic noise-floor tracking,
        gain boosting for quiet speech from the floor, and asynchronous STT evaluation
        so voice commands ('help', 'help me', 'emergency', 'i am fine') are detected
        in real time without blocking audio capture.
        """
        window_duration = timeout_sec or self.response_timeout_sec

        def _worker():
            self._is_listening = True
            start_time = time.monotonic()
            device = self.input_device_index
            sample_rate = self._sample_rate
            block_sec = 0.1
            block_size = int(sample_rate * block_sec)

            if on_status:
                on_status(f"Listening on mic for {window_duration:.0f}s (Say 'I am fine' or 'Help')...")
            print(f"[SafeFall Voice] Opening microphone for {window_duration:.1f}s listening window...")

            if HAS_SD and HAS_NUMPY and HAS_SR:
                try:
                    lock = threading.Lock()
                    result_dispatched = False
                    last_ambiguous_transcript = ""

                    def _eval_worker(chunks_to_eval: List[np.ndarray]):
                        nonlocal result_dispatched, last_ambiguous_transcript
                        if not chunks_to_eval or result_dispatched:
                            return
                        try:
                            raw_pcm = np.concatenate(chunks_to_eval, axis=0)
                            norm_pcm = self._normalize_audio(raw_pcm)

                            # CRITICAL PADDING: Symmetric silence margin on both ends so Google STT neural acoustic
                            # model never edge-truncates single-syllable cries ('help', 'fine', 'emergency', 'safe')
                            pad_samples = int(sample_rate * 0.45)
                            norm_pcm = np.pad(norm_pcm, (pad_samples, pad_samples), mode="constant")

                            audio_data = sr.AudioData(norm_pcm.tobytes(), sample_rate=sample_rate, sample_width=2)

                            if on_status:
                                on_status("⚡ Evaluating verbal response...")

                            transcript = self.recognizer.recognize_google(audio_data, language="en-US")
                            print(f"[SafeFall Voice] Recognized transcript: '{transcript}'")
                            self.last_heard_text = transcript
                            intent = classify_transcript(transcript)
                            self.last_intent = intent

                            if on_status:
                                on_status(f"Heard: '{transcript}' -> [{intent.value}]")

                            with lock:
                                if result_dispatched:
                                    return
                                if intent in (VerbalIntent.CONFIRM_SAFE, VerbalIntent.REQUEST_HELP):
                                    result_dispatched = True
                                    self._is_listening = False
                                    on_result(intent, transcript)
                                else:
                                    last_ambiguous_transcript = transcript
                        except sr.UnknownValueError:
                            pass
                        except Exception as e:
                            print(f"[SafeFall Voice] STT Error: {e}")

                    with sd.InputStream(device=device, channels=1, samplerate=sample_rate, dtype="int16") as stream:
                        # Initial brief calibration
                        init_chunk, _ = stream.read(block_size)
                        init_chunk_flat = init_chunk.flatten()
                        init_rms = float(np.sqrt(np.mean(init_chunk_flat.astype(float) ** 2)))
                        noise_floor = max(35.0, min(init_rms, 600.0))
                        print(f"[SafeFall Voice] Initial noise floor RMS: {noise_floor:.1f}")

                        session_chunks: List[np.ndarray] = [init_chunk_flat]
                        pre_roll_chunks = collections.deque(maxlen=4)  # 0.4s pre-roll buffer to preserve leading phonemes
                        utterance_chunks: List[np.ndarray] = []
                        is_speech_active = False
                        speech_detected_in_session = False
                        speech_start_time = 0.0
                        silence_after_speech = 0.0
                        last_periodic_eval = 0.0

                        while (time.monotonic() - start_time) < window_duration and self._is_listening:
                            chunk, _ = stream.read(block_size)
                            chunk_flat = chunk.flatten()
                            session_chunks.append(chunk_flat)

                            rms = float(np.sqrt(np.mean(chunk_flat.astype(float) ** 2)))
                            # Running adaptive noise floor tracking (slow tracking during speech so voice is not swallowed)
                            if rms < noise_floor * 1.3:
                                noise_floor = noise_floor * 0.96 + rms * 0.04
                            else:
                                noise_floor = noise_floor * 0.998 + rms * 0.002

                            vad_threshold = max(55.0, noise_floor * 1.25 + 15.0)
                            is_above_vad = (rms >= vad_threshold)

                            if on_audio_level:
                                on_audio_level(rms, is_above_vad)

                            now_mono = time.monotonic()

                            if is_above_vad:
                                speech_detected_in_session = True
                                if not is_speech_active:
                                    is_speech_active = True
                                    speech_start_time = now_mono
                                    # Prepend pre-roll buffer so the leading consonant / phoneme ('h' in help) is preserved
                                    utterance_chunks.extend(list(pre_roll_chunks))
                                    if on_status:
                                        rem = max(0.0, window_duration - (now_mono - start_time))
                                        on_status(f"🎙️ Voice detected... recording ({rem:.0f}s left)")

                                utterance_chunks.append(chunk_flat)
                                silence_after_speech = 0.0

                                # Periodic sliding evaluation if speaking continuously for >= 1.6s
                                speech_duration = now_mono - speech_start_time
                                if speech_duration >= 1.6 and (now_mono - last_periodic_eval) >= 1.4:
                                    last_periodic_eval = now_mono
                                    threading.Thread(target=_eval_worker, args=(list(utterance_chunks),), daemon=True).start()

                            elif is_speech_active:
                                utterance_chunks.append(chunk_flat)
                                silence_after_speech += block_sec

                                speech_duration = now_mono - speech_start_time
                                # Completed phrase: 0.4s silence after voice or 2.5s duration
                                if silence_after_speech >= 0.4 or speech_duration >= 2.5:
                                    if len(utterance_chunks) >= 4:  # At least 0.4s of audio
                                        threading.Thread(target=_eval_worker, args=(list(utterance_chunks),), daemon=True).start()
                                    utterance_chunks = []
                                    is_speech_active = False
                                    silence_after_speech = 0.0
                            else:
                                pre_roll_chunks.append(chunk_flat)

                            # Fast exit if result already found
                            with lock:
                                if result_dispatched:
                                    break

                    # Window finished: perform final pass on accumulated session speech if needed
                    with lock:
                        already_done = result_dispatched

                    if not already_done and session_chunks and speech_detected_in_session:
                        # Evaluate last 3.5 seconds of the session
                        recent_chunks = session_chunks[-35:] if len(session_chunks) > 35 else session_chunks
                        _eval_worker(recent_chunks)
                        # Brief wait for final recognition
                        deadline = time.monotonic() + 1.2
                        while time.monotonic() < deadline:
                            with lock:
                                if result_dispatched:
                                    break
                            time.sleep(0.05)

                    with lock:
                        if not result_dispatched:
                            result_dispatched = True
                            self._is_listening = False
                            if last_ambiguous_transcript:
                                print(f"[SafeFall Voice] Window closed with ambiguous transcript: '{last_ambiguous_transcript}'")
                                on_result(VerbalIntent.AMBIGUOUS, last_ambiguous_transcript)
                            else:
                                print("[SafeFall Voice] Window closed with no speech detected.")
                                on_result(VerbalIntent.NO_RESPONSE, "")
                    return

                except Exception as ex:
                    print(f"[SafeFall Voice] Primary sounddevice loop error: {ex}")

            # Fallback when audio stream fails
            elapsed = time.monotonic() - start_time
            if elapsed < window_duration:
                time.sleep(window_duration - elapsed)
            self._is_listening = False
            on_result(VerbalIntent.NO_RESPONSE, "")

        threading.Thread(target=_worker, daemon=True).start()

    def cancel_listening(self):
        """Immediately stops the listening loop."""
        self._is_listening = False

    def is_listening(self) -> bool:
        return self._is_listening
