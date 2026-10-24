# SafeFall Android Setup & Run Guide

## 1. Prerequisites
- **Flutter SDK**: 3.19+ installed and on PATH
- **Android Studio / Android SDK**: API Level 34 with Android Build Tools
- **Java**: JDK 17
- **Target Device / Emulator**: Physical Android phone (recommended for camera & mic) or Android Emulator with Camera forwarding enabled.

## 2. Android Configuration
1. **Permissions** (configured in `android/app/src/main/AndroidManifest.xml`):
   - `android.permission.CAMERA`
   - `android.permission.RECORD_AUDIO`
   - `android.permission.INTERNET`
   - `android.permission.WAKE_LOCK`
   - `android.permission.FOREGROUND_SERVICE`
   - `android.permission.CALL_PHONE` (for emergency call action)
2. **Firebase Setup**:
   - Place `google-services.json` into `flutter_apps/android/app/`.

## 3. Building and Running

### 3.1 Launch on Connected Android Device
```bash
cd safefall/flutter_apps
flutter run -d android
```

### 3.2 Building APK for Distribution
```bash
flutter build apk --release
```
The resulting APK is generated at:
`flutter_apps/build/app/outputs/flutter-apk/app-release.apk`
