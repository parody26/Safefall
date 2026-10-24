import sounddevice as sd
import numpy as np
import speech_recognition as sr
import time

print("=" * 60)
print("SafeFall Audio & Speech Recognition Diagnostic")
print("=" * 60)

devices = sd.query_devices()
def_in = sd.default.device[0]
print(f"Default input device index: {def_in}")
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        star = " -> [DEFAULT]" if i == def_in else ""
        print(f"  [{i}] {d['name']} ({d['max_input_channels']} in){star}")

print("\nCapturing 2.0 seconds of room audio...")
data = sd.rec(int(16000 * 2.0), samplerate=16000, channels=1, dtype="int16")
sd.wait()
rms = float(np.sqrt(np.mean(data.astype(float) ** 2)))
peak = int(np.max(np.abs(data)))
print(f"Captured: 32000 samples | RMS = {rms:.2f} | Peak = {peak}")

rec = sr.Recognizer()
aud = sr.AudioData(data.tobytes(), 16000, 2)
print("Testing cloud STT connectivity...")
try:
    txt = rec.recognize_google(aud)
    print(f"Transcribed: '{txt}'")
except sr.UnknownValueError:
    print("STT responded: [UnknownValueError] (Quiet room or no recognizable English words - this is NORMAL for background noise)")
except Exception as e:
    print(f"STT Error: {type(e).__name__}: {e}")

print("=" * 60)
