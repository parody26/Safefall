# SafeFall System Architecture

## 1. High-Level System Architecture
SafeFall is engineered as a privacy-first, edge-intelligent fall detection and rapid emergency response ecosystem.

```mermaid
flowchart TB
    subgraph EdgeMonitoring["Edge Monitoring Clients (Continuous Vision & Audio)"]
        subgraph WinMon["Windows Monitor"]
            WCam[Webcam / CCTV RTSP] --> WCV[Pose Estimation & Temporal Fall Detector]
            WCV --> WSM[Emergency State Machine]
            WSM --> WAudio[Local Alarm (Beep-Beep)]
            WSM --> WVoice[Local Speech-to-Text & Intent Classifier]
        end
        subgraph MobileMon["Mobile Monitors (Android & iOS)"]
            MCam[Device Camera] --> MCV[On-Device Pose Estimation & Fall Detector]
            MCV --> MSM[Emergency State Machine]
            MSM --> MAudio[Local Audio Alert]
            MSM --> MVoice[Microphone Speech Recognition]
        end
    end

    subgraph Cloud["Unified Backend Infrastructure (Firebase / Edge API)"]
        Auth[Firebase Auth - JWT Identity]
        Firestore[Cloud Firestore - Incidents, Devices, Families]
        Storage[Firebase Storage - Authenticated Evidence Vault]
        FCM[Firebase Cloud Messaging / APNs]
        Functions[Cloud Functions - Escalation & Audit Trail]
    end

    subgraph FamilyApps["Family Safety Clients (Cross-Platform Flutter UI)"]
        FWin[Windows Family App]
        FAndroid[Android Family App]
        FiOS[iPhone / iOS Family App]
    end

    WSM -- Incident Upload --> Cloud
    MSM -- Incident Upload --> Cloud
    Cloud -- Push Notifications --> FamilyApps
    FamilyApps -- Acknowledge / Resolve / Emergency Call --> Cloud
    Cloud -- State Sync --> EdgeMonitoring
```

## 2. Monorepo Structural Organization
```
safefall/
├── README.md
├── .gitignore
├── .env.example
├── docs/                      # Full technical & platform documentation
│   ├── architecture.md
│   ├── platform_support.md
│   ├── setup_windows.md
│   ├── setup_android.md
│   ├── setup_ios.md
│   ├── firebase_setup.md
│   ├── security.md
│   ├── testing.md
│   ├── deployment.md
│   └── limitations.md
├── shared/                    # Formal specs shared across all clients
│   ├── detection_spec.md
│   ├── incident_schema.md
│   ├── state_machine.md
│   ├── api_contract.md
│   └── test_scenarios.md
├── windows_monitor/           # Windows Desktop Monitoring Application
│   ├── src/
│   │   ├── camera/
│   │   ├── pose/
│   │   ├── detection/
│   │   ├── alarm/
│   │   ├── speech/
│   │   ├── backend/
│   │   └── ui/
│   ├── tests/
│   └── requirements.txt
├── flutter_apps/              # Unified Cross-Platform Flutter App (Android, iOS, Windows)
│   ├── lib/
│   │   ├── core/
│   │   ├── auth/
│   │   ├── family/
│   │   ├── devices/
│   │   ├── monitoring/
│   │   ├── incidents/
│   │   ├── notifications/
│   │   └── settings/
│   ├── android/
│   ├── ios/
│   ├── windows/
│   ├── test/
│   └── pubspec.yaml
├── backend/                   # Backend Security Rules and Cloud Functions
│   ├── functions/
│   ├── firestore.rules
│   ├── firestore.indexes.json
│   └── storage.rules
└── test_data/                 # Synthetic & prerecorded test sequences
    └── README.md
```

## 3. Subsystem Breakdown
### 3.1 Edge Computer Vision & Posture Classification
Frames are sampled at a configurable frequency (default 5–10 Hz, 100–200ms per frame). MediaPipe or YOLO-pose keypoints yield 33 anatomical landmarks. Geometric heuristics evaluate torso angle against the vertical axis, bounding envelope aspect ratio, and relative hip-to-ground elevation.

### 3.2 Temporal Fall Analysis
A rolling window of observations is evaluated for rapid velocity vectors:
- Rapid decrease in vertical hip coordinate: $\Delta Y_{hip} / \Delta t > v_{threshold}$
- Rapid rotational tilt: $\Delta \theta_{torso} / \Delta t > \omega_{threshold}$
- Posture settling in `LYING` for $> 1.5$ seconds without immediate recovery.

### 3.3 Emergency State Machine & Verbal Verification
The state machine implements a two-stage filter:
1. **Suspected Fall**: 3-second grace window to confirm the subject did not immediately stand up.
2. **Local Alarm & Verbal Verification**: 10-second window with repeating acoustic alarm. Speech recognition evaluates verbal intent (`CONFIRM_SAFE`, `REQUEST_HELP`, `AMBIGUOUS`).
3. **Escalation**: Escalates immediately on `REQUEST_HELP` or timeout of `NO_RESPONSE` / `SPEECH_ERROR`.

### 3.4 Cross-Platform Family Response UI
Built with Flutter, the family safety application operates identically across Android, iOS, and Windows Desktop. It provides real-time emergency banners, evidence review, single-tap acknowledgment, resolution note logging, and direct emergency dialer integration.
