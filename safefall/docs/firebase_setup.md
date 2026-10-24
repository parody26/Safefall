# SafeFall Firebase Setup & Configuration Guide

SafeFall uses Firebase for identity, state synchronization, secure incident vaulting, and family notification dispatch.

## 1. Firebase Project Creation
1. Go to [Firebase Console](https://console.firebase.google.com/) and click **Add project**.
2. Name the project `safefall-app` (or your chosen identifier).
3. Enable **Google Analytics** (optional) and create project.

## 2. Authentication Setup
1. In the Firebase Console, navigate to **Build > Authentication > Sign-in method**.
2. Enable **Email/Password**.
3. Under **Templates**, configure the password reset and email verification email templates with your organization name.

## 3. Cloud Firestore Provisioning
1. Navigate to **Build > Firestore Database**.
2. Click **Create database**, choose **Production mode**, and select a region close to your primary deployment (e.g. `asia-south1` or `us-central1`).
3. Deploy the database rules and indexes provided in `backend/firestore.rules` and `backend/firestore.indexes.json` using Firebase CLI:
   ```bash
   firebase deploy --only firestore:rules,firestore:indexes
   ```

## 4. Firebase Storage Provisioning
1. Navigate to **Build > Storage**.
2. Click **Get started**, choose **Production mode**.
3. Deploy storage rules:
   ```bash
   firebase deploy --only storage:rules
   ```

## 5. Firebase Cloud Messaging (FCM) & APNs Setup
1. Under **Project Settings > Cloud Messaging**:
   - Generate Web Push Certificates / Key pair if using web.
   - For **Apple Push Notification service (APNs)**: Upload your Apple Developer APNs Authentication Key (`.p8` file), along with your Team ID and Key ID.
2. For Android: Download `google-services.json` and place in `flutter_apps/android/app/google-services.json`.
3. For iOS: Download `GoogleService-Info.plist` and place in `flutter_apps/ios/Runner/GoogleService-Info.plist`.

## 6. Cloud Functions Deployment
1. Install dependencies and deploy:
   ```bash
   cd backend/functions
   npm install
   firebase deploy --only functions
   ```

## 7. Zero-Cloud Mock Mode (Development & Testing)
SafeFall includes a self-contained local backend (`windows_monitor/src/backend/mock_backend.py` / `test_data/local_api.py`) allowing full end-to-end testing of authentication, device pairing, incident generation, and family alerts without needing live cloud credentials!
