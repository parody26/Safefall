# SafeFall Cross-Platform Matrix & Platform Support

SafeFall explicitly supports all six (6) platform and application combinations.

| Application Role | Windows (Desktop) | Android | iPhone / iOS |
|---|---|---|---|
| **Monitoring Application** | **Required** (Python/OpenCV/PySide6 desktop app with camera, audio alarm, speech recognition, state machine) | **Required** (Flutter app with camera preview, on-device pose estimation, audio alarm, speech-to-text) | **Required** (Flutter app with AVFoundation camera, CoreML/MediaPipe pose estimation, AVAudioSession alarm, speech) |
| **Family Safety Application** | **Required** (Flutter desktop with real-time stream, push banner, acknowledge/resolve, emergency call) | **Required** (Flutter mobile with FCM notifications, emergency dialer, evidence review, full family management) | **Required** (Flutter mobile with APNs/FCM critical notifications, emergency call, biometric auth, evidence view) |

## 1. Windows Implementation Details
- **Monitoring Application**:
  - Implementation: Python 3 with OpenCV for video feed acquisition, MediaPipe / YOLO Pose for landmark extraction, PySide6 / Qt desktop interface, Windows Speech API / SpeechRecognition, and `winsound` for acoustic alarm generation.
  - Platform Capabilities: Direct USB webcam or RTSP IP camera support; high-framerate processing; unrestricted desktop compute.
  - Limitations: Device sleep/hibernation suspends camera capture; Windows Power Management should be configured to "High Performance / Never Sleep".
- **Family Application**:
  - Implementation: Flutter Windows Desktop.
  - Platform Capabilities: Native Win32 windowing, desktop notification toasts, side-by-side split pane responsive layout.

## 2. Android Implementation Details
- **Monitoring Application**:
  - Implementation: Flutter with `camera` package, on-device ML inference, `audioplayers` / native tone generator, `speech_to_text`.
  - Permissions: `android.permission.CAMERA`, `android.permission.RECORD_AUDIO`, `android.permission.WAKE_LOCK`, `android.permission.FOREGROUND_SERVICE`.
  - Android Lifecycle: Foreground Service with persistent notification required to maintain camera acquisition when app is not in fullscreen.
- **Family Application**:
  - Implementation: Flutter Android with Firebase Cloud Messaging (FCM), high-priority notification channels, native intent dialer (`url_launcher: tel:112`).

## 3. iPhone / iOS Implementation Details
- **Monitoring Application**:
  - Implementation: Flutter iOS targeting iOS 16+.
  - Permissions: `NSCameraUsageDescription`, `NSMicrophoneUsageDescription`, `NSSpeechRecognitionUsageDescription`.
  - iOS Lifecycle Requirements: iOS actively restricts continuous camera capture in background mode for privacy reasons. The monitoring app is designed for **kiosk/foreground mode** with screen kept awake (`wakelock_plus`) when active on a bedside or living room stand/dock.
- **Family Application**:
  - Implementation: Flutter iOS with Apple Push Notification service (APNs) + FCM. Supports critical alerts, badge counters, and native dialer.

## 4. Hardware, Credential & Signing Requirements
- **Windows**: No developer account required.
- **Android**: Google Play Services for FCM; Android Studio / SDK for compilation.
- **iOS**: Apple Developer Account ($99/year) required for code signing, provisioning profiles, and APNs critical alert entitlements. Physical iPhone required for camera/APNs testing (iOS Simulator does not have a camera or APNs push capability).
