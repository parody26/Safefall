# SafeFall Production Deployment & Operations Guide

## 1. System Components Deployment Overview
- **Edge Monitors**:
  - Windows: Installed via standalone MSI / executable package or run as an auto-starting system service (`windows_monitor/src/main.py`).
  - Android: Deployed via Google Play Internal Testing / APK sideloading with Foreground Service enabled.
  - iOS: Deployed via Apple TestFlight with camera privacy entitlements.
- **Backend & Cloud Infrastructure**:
  - Google Cloud Firestore (Managed NoSQL Database).
  - Firebase Cloud Functions (Serverless Node.js 18+).
  - Firebase Cloud Messaging & APNs Gateway.
- **Family Safety Applications**:
  - Desktop: Windows MSIX or standalone binary.
  - Mobile: Android App Bundle (`.aab`) and iOS App Store archive (`.ipa`).

## 2. CI/CD Pipeline Recommendations
- Automated GitHub Actions workflow testing:
  1. Python unit tests & linting on Windows runner.
  2. Flutter unit & widget tests across Linux / Windows runners.
  3. Firestore security rules unit tests using `@firebase/rules-unit-testing`.
