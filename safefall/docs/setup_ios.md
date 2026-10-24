# SafeFall iOS / iPhone Setup & Build Guide

## 1. Prerequisites
- **macOS Workstation**: macOS Sonoma 14+ with Xcode 15+ installed.
- **Apple Developer Account**: Required for code signing, physical camera access entitlements, and APNs critical push alerts ($99/yr).
- **Physical iPhone**: Running iOS 16.0 or newer. *(Note: iOS Simulator does not provide physical camera video or APNs push notifications).*

## 2. iOS Configuration Requirements
1. **Info.plist Privacy Keys** (located in `flutter_apps/ios/Runner/Info.plist`):
   - `NSCameraUsageDescription`: "SafeFall requires continuous camera access in the foreground to detect falls and ensure elder safety."
   - `NSMicrophoneUsageDescription`: "SafeFall requires microphone access during emergency alarms to listen for verbal confirmation of safety or requests for help."
   - `NSSpeechRecognitionUsageDescription`: "SafeFall uses speech recognition to classify verbal responses after a detected fall."
2. **Push Notification Capabilities**:
   - In Xcode under **Signing & Capabilities**:
     - Add **Push Notifications**.
     - Add **Background Modes** -> check **Remote notifications**.
     - For high-priority siren alarms: Apply for Apple's **Critical Alerts Entitlement** (`com.apple.developer.usernotifications.critical-alerts`).
3. **Firebase Configuration**:
   - Place `GoogleService-Info.plist` into `flutter_apps/ios/Runner/GoogleService-Info.plist`.

## 3. Building and Running

### 3.1 Install CocoaPods
```bash
cd safefall/flutter_apps/ios
pod install
```

### 3.2 Launch on Physical iPhone
```bash
cd ..
flutter run -d <your-iphone-device-id>
```

### 3.3 Archive and Distribute via TestFlight
```bash
flutter build ipa --export-method app-store
```
