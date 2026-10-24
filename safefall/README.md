# 🛡️ SafeFall
### Cross-Platform AI Fall Detection, Family Safety and Emergency Response Platform

SafeFall is a privacy-first, edge-intelligent fall detection and rapid emergency response platform designed for elder care and vulnerable family members.

---

## 📱 Supported Applications & Platforms Matrix

SafeFall provides 6 distinct application/platform operational combinations:

| Application Target | Windows (Desktop) | Android Phone / Tablet | iPhone / iOS |
|---|---|---|---|
| **Monitoring Application** | ✅ **Active & Tested** (`windows_monitor`) Python 3 + OpenCV + Biomechanical AI + Winsound + Speech + Desktop GUI | ✅ **Implemented** (`flutter_apps`) Camera preview + Pose AI + Alarm tone + SpeechRecognizer | ✅ **Implemented** (`flutter_apps`) AVFoundation Camera + Pose AI + AVAudioSession + SFSpeech |
| **Family Safety Application** | ✅ **Implemented** (`flutter_apps`) Desktop Responsive Shell + Realtime stream + Alerts + Evidence | ✅ **Implemented** (`flutter_apps`) Mobile Responsive Shell + FCM Notifications + Emergency Dialer | ✅ **Implemented** (`flutter_apps`) Mobile Shell + APNs Critical Alerts + Evidence Vault + 112 Dialer |

---

## 🏛️ System Architecture & Workflow

```
[Camera Source (Webcam/Mobile)]
              │
              ▼
   [Biomechanical Pose Extraction]
   (Torso Angle θ, Aspect Ratio AR, Knee Flexion α, Hip Height Y)
              │
              ▼
   [Temporal Fall Detector] ── (Evaluates velocity ΔY/Δt & angular tilt over rolling window)
              │ (Rapid transition to LYING)
              ▼
   [Emergency State Machine]
              │
              ├── State B: SUSPECTED_FALL (3.0s monotonic countdown)
              │      └── Person stands up? ──> CANCELLED (Return to NORMAL)
              │
              ├── State C & D: LOCAL_ALARM & VOICE_PROCESSING
              │      ├── Repeating Beep-Beep Alarm sounds
              │      ├── Prompts "Are you okay?" & opens microphone for 10.0s
              │      ├── User says "I am fine" ──> Alarm Stops, Marked False Alarm
              │      ├── User says "Help me" ──> Immediate Escalation
              │      └── Silence / Timeout ──> Safe Failsafe Escalation
              │
              ▼
   [State G: EMERGENCY_PUBLISHED] ──> Cloud Firestore & Multicast Push (FCM / APNs)
              │
              ▼
   [Family Safety Application]
              ├── Critical Emergency Alert Banner
              ├── Encrypted Keyframe Evidence Vault
              ├── Single-Tap "Acknowledge" (Logs timestamp & member UID)
              ├── Physical Check & "Resolve" (Logs resolution note)
              └── Immediate Emergency Dial ("Call 112" / Regional Dispatch)
```

---

## 🚀 Quick Start Guide

### 1. Launch the System (3 Components)

#### Component A: SafeFall Shared Backend (Port 8080)
```powershell
cd safefall
python backend/server.py
# Or double-click: run_backend.bat
```

#### Component B: Windows AI Fall Monitoring Station (24 FPS Camera + Voice AI)
```powershell
cd safefall/windows_monitor
python src/main.py
# Or double-click: run_monitor.bat
```

#### Component C: SafeFall Family Safety Application (Real-Time Push Alerts & Evidence)
```powershell
cd safefall
python family_app/main.py
# Or double-click: run_family_app.bat
```

### 2. Automated Test & Benchmark Suite
```powershell
cd safefall/windows_monitor
python -m unittest discover tests -v
python tests/benchmark_detector.py
```

### 3. Cross-Platform Mobile Family & Monitor (Flutter)
```bash
cd safefall/flutter_apps
flutter run -d android   # Android Phone / Tablet
flutter run -d ios       # iPhone / iPad
flutter run -d windows   # Windows Desktop Shell
```

---

## 🧪 3-Minute Hackathon Demonstration Script

1. **Launch Windows Monitoring Station in Simulation Mode**:
   ```powershell
   cd safefall/windows_monitor
   python src/main.py --simulation
   ```
   - Show the live posture badge reading `STANDING` (Confidence: 94%, Torso Angle: 8.2°).
   - Point out hardware status indicators: Camera `ONLINE`, Mic `READY`, Cloud Sync `CONNECTED`.

2. **Trigger a Fall**:
   - Click **🚨 Trigger Test Fall (3s -> 10s)**.
   - The status bar immediately transitions to amber: **⚠️ SUSPECTED FALL DETECTED (3.0s grace)**.
   - Show the 3.0s countdown running on a monotonic timer.

3. **Verbal Verification & Acoustic Alarm**:
   - At $t = 3.0$s, the local acoustic repeating **beep-beep alarm** sounds on the device speaker.
   - The microphone activates with a 10.0s countdown: **🚨 EMERGENCY ALARM ACTIVE — Say 'I am fine' or 'Help'**.

4. **Scenario A — False Alarm Cancellation**:
   - Click **🗣️ Simulate Reassurance ('I am fine')**.
   - The alarm immediately stops. The system logs an audit event and safely restores monitoring without disturbing family members.

5. **Scenario B — Urgent Help Cry Escalation**:
   - Trigger fall again and click **🆘 Simulate Help Request ('Help me!')**.
   - The system escalates immediately without waiting for the timeout!
   - State advances to **EMERGENCY_PUBLISHED** and persists the incident to the cloud backend.

6. **Inspect in Family Safety Application**:
   - Launch `flutter_apps`.
   - The family dashboard flashes a high-contrast emergency banner: **🚨 ACTIVE FALL EMERGENCY DETECTED**.
   - Tap **View Evidence & Actions**: Inspect the keyframe, posture transition, confidence, and audit timeline.
   - Tap **Acknowledge**: Updates the record with the acknowledging caregiver's name and timestamp.
   - Tap **Resolve**: Enter note *"Checked Grandma, she is safe in bed"*. The emergency is closed and normal monitoring resumes.
   - Demonstrate the **Call 112** button launching the native dialer.

---

## 🔒 Privacy & Security Commitments

1. **No Continuous Cloud Video**: Ordinary camera frames are strictly processed in volatile RAM and immediately discarded.
2. **Ephemeral Evidentiary Snapshots**: Only when a fall is verified does the system retain a keyframe in an encrypted storage bucket accessible only to authenticated family members.
3. **Role-Based Security**: Backed by production-grade Firestore Security Rules (`backend/firestore.rules`) enforcing Administrator, Caregiver, and Viewer role scopes.
4. **No Server Secrets on Clients**: Client applications never contain Firebase Admin service account keys.

---

## 📋 Honest Platform Disclosures & Requirements

- **Windows**: Fully operational and tested locally on Windows 11 with Python 3.14, OpenCV, and native Tkinter desktop interface.
- **Android**: Flutter codebase and Android manifest with permissions (`CAMERA`, `RECORD_AUDIO`, `CALL_PHONE`, `FOREGROUND_SERVICE`) are fully implemented. Physical Android device recommended for live camera and FCM push testing.
- **iOS**: Flutter codebase and iOS `Info.plist` with privacy keys (`NSCameraUsageDescription`, `NSMicrophoneUsageDescription`, `NSSpeechRecognitionUsageDescription`) and URL schemes are implemented. An active Apple Developer Account ($99/year) and physical iPhone are required for building with code signing and APNs push capabilities. Continuous background camera monitoring is subject to Apple iOS foreground restrictions (kiosk dock mode recommended).
