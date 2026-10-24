"""
SafeFall Standalone Interactive Voice Input & Alarm Cancellation Test.
Runs directly on the user's physical microphone, measures live RMS volume,
prompts the user, transcribes via Google STT, and tests intent classification.
"""

import sys
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from src.speech.speech_service import SpeechService, classify_transcript, VerbalIntent

def run_diagnostic():
    print("=" * 65)
    print("   SafeFall — Real-Time Live Microphone Diagnostic")
    print("=" * 65)

    # 1. Hardware Detection
    devices = SpeechService.list_input_devices()
    print("Available Audio Input Microphones:")
    for d in devices:
        def_tag = " [ACTIVE DEFAULT]" if d['is_default'] else ""
        print(f"  [{d['index']}] {d['name']}{def_tag}")

    default_dev = next((d for d in devices if d['is_default']), None) or (devices[0] if devices else None)
    if not default_dev:
        print("[ERROR] No microphone detected on this system!")
        return

    print(f"\nTarget Microphone: [{default_dev['index']}] {default_dev['name']}")
    print("=" * 65)

    # 2. Ambient Noise Calibration (1.0 sec)
    print("\n[Step 1/3] Calibrating room ambient noise (1.0s)... Please remain quiet...")
    block_samples = 16000
    ambient_audio = sd.rec(block_samples, samplerate=16000, channels=1, dtype="int16", device=default_dev['index'])
    sd.wait()
    ambient_rms = float(np.sqrt(np.mean(ambient_audio.astype(float) ** 2)))
    vad_threshold = max(35.0, min(ambient_rms * 1.25 + 10.0, ambient_rms + 45.0))
    print(f"  -> Ambient Noise Floor RMS: {ambient_rms:.1f}")
    print(f"  -> Adaptive Voice VAD Threshold: {vad_threshold:.1f}")

    # 3. Prompt User
    speech = SpeechService(response_timeout_sec=6.0, input_device_index=default_dev['index'])
    print("\n[Step 2/3] Speaking prompt: 'SafeFall alert. Are you okay?'")
    speech.prompt_verbal_check("SafeFall alert. Are you okay? Say help me or I am fine.")
    time.sleep(3.2) # allow prompt to finish cleanly

    # 4. Live Recording & Metering (6.0s)
    print("\n[Step 3/3] 🎙️ LISTENING NOW (6.0 seconds)!")
    print("  👉 To test EMERGENCY CRY: Say 'Help me', 'Help', or 'Emergency'")
    print("  👉 To test SAFE CANCELLATION: Say 'I am fine', 'I am okay', or 'Stop'")
    print("  (Speak at normal volume towards your microphone)\n")

    captured_chunks = []
    chunk_size = int(16000 * 0.25)
    speech_frames = []

    with sd.InputStream(device=default_dev['index'], channels=1, samplerate=16000, dtype="int16") as stream:
        for i in range(24): # 24 * 0.25s = 6.0 seconds
            chunk, _ = stream.read(chunk_size)
            rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
            speech_frames.append(chunk.flatten())

            # Visual volume bar
            bars = min(30, int(rms / 12.0))
            bar_str = "█" * bars + "░" * (30 - bars)
            status = " [VOICE DETECTED]" if rms >= vad_threshold else " [ambient]"
            sys.stdout.write(f"\r  Audio Level: [{bar_str}] RMS: {rms:5.1f}{status}  ")
            sys.stdout.flush()

    print("\n\nEvaluating recorded speech with Google Speech Recognition...")
    raw_pcm = np.concatenate(speech_frames, axis=0)
    norm_pcm = speech._normalize_audio(raw_pcm)

    # Apply minimum 1.5s padding so single-word cries ('help', 'emergency') succeed
    min_samples = int(16000 * 1.5)
    if len(norm_pcm) < min_samples:
        pad_needed = min_samples - len(norm_pcm)
        norm_pcm = np.pad(norm_pcm, (pad_needed // 2, pad_needed - pad_needed // 2), mode="constant")

    audio_data = sr.AudioData(norm_pcm.tobytes(), sample_rate=16000, sample_width=2)

    rec = sr.Recognizer()
    try:
        transcript = rec.recognize_google(audio_data, language="en-US")
        print(f"\n>>> RECOGNIZED TRANSCRIPT: \"{transcript}\"")
        intent = classify_transcript(transcript)
        print(f">>> CLASSIFIED INTENT:    {intent.value}")

        if intent == VerbalIntent.CONFIRM_SAFE:
            print("\n" + "=" * 65)
            print("  ✅ TEST RESULT: PASS (CONFIRMED SAFE)")
            print("  The microphone works properly and this phrase WILL SILENCE THE ALARM!")
            print("=" * 65)
            speech.speak_confirmation("Microphone verified. Alarm successfully cancelled.")
        elif intent == VerbalIntent.REQUEST_HELP:
            print("\n" + "=" * 65)
            print("  🚨 TEST RESULT: PASS (REQUEST HELP)")
            print("  The microphone works properly and detected your urgent help cry!")
            print("  This immediately triggers alarm escalation and dispatches to Family App!")
            print("=" * 65)
        else:
            print("\n" + "=" * 65)
            print(f"  ⚠️ TEST RESULT: AMBIGUOUS INTENT ('{transcript}')")
            print("  The microphone heard your voice! Recognized keywords: 'help', 'help me', 'emergency', 'i am fine'.")
            print("=" * 65)

    except sr.UnknownValueError:
        print("\n" + "=" * 65)
        print("  ❌ Google STT: Unintelligible audio or room too quiet.")
        print("  Tips:")
        print("  1. Make sure your Logitech C270 microphone is not muted in Windows Settings.")
        print("  2. Speak slightly louder and closer to the microphone.")
        print("  3. Run the test again and speak 'I am fine' during the 5.0s window.")
        print("=" * 65)
    except Exception as e:
        print(f"\n[ERROR] Cloud STT error: {e}")

if __name__ == "__main__":
    run_diagnostic()
