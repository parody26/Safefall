# SafeFall Unified Incident & Data Schema Specification

## 1. Cloud Firestore Collection Layout

### 1.1 `users/{uid}`
```json
{
  "uid": "usr_948a7b1c",
  "displayName": "Sarah Jenkins",
  "email": "sarah.jenkins@example.com",
  "createdAt": "2026-10-23T18:00:00Z",
  "accountStatus": "active",
  "fcmTokens": ["fcm_token_device_abc123"]
}
```

### 1.2 `families/{familyId}`
```json
{
  "familyId": "fam_481029",
  "name": "Jenkins Family Residence",
  "description": "Home care safety monitoring for Grandma Alice",
  "createdBy": "usr_948a7b1c",
  "createdAt": "2026-10-23T18:05:00Z",
  "settings": {
    "emergencyDialNumber": "112",
    "evidenceRetentionDays": 30,
    "alarmAudibleCountdownSec": 3,
    "voiceResponseWindowSec": 10
  }
}
```

### 1.3 `memberships/{membershipId}` (`{familyId}_{uid}`)
```json
{
  "membershipId": "fam_481029_usr_948a7b1c",
  "familyId": "fam_481029",
  "uid": "usr_948a7b1c",
  "role": "admin", // "admin" | "caregiver" | "viewer"
  "joinedAt": "2026-10-23T18:05:00Z",
  "membershipStatus": "active",
  "notificationPreferences": {
    "pushEnabled": true,
    "soundEnabled": true,
    "criticalAlerts": true
  }
}
```

### 1.4 `invitations/{invitationId}`
```json
{
  "invitationId": "inv_782910",
  "familyId": "fam_481029",
  "familyName": "Jenkins Family Residence",
  "inviteCode": "SAFE-8942",
  "createdBy": "usr_948a7b1c",
  "role": "caregiver",
  "createdAt": "2026-10-23T18:10:00Z",
  "expiresAt": "2026-10-25T18:10:00Z",
  "usageLimit": 5,
  "usageCount": 1,
  "revokedAt": null
}
```

### 1.5 `devices/{deviceId}`
```json
{
  "deviceId": "dev_win_01",
  "familyId": "fam_481029",
  "deviceName": "Living Room Monitor",
  "platform": "windows", // "windows" | "android" | "ios"
  "locationLabel": "Living Room (Main Area)",
  "createdAt": "2026-10-23T18:15:00Z",
  "lastSeenAt": "2026-10-23T21:59:00Z",
  "monitoringStatus": "monitoring_active", // "idle" | "monitoring_active" | "suspected_fall" | "alarm_sounding"
  "registrationStatus": "paired",
  "configuration": {
    "frameIntervalMs": 200,
    "confidenceThreshold": 0.55,
    "voiceEnabled": true,
    "evidenceCaptureEnabled": true
  }
}
```

### 1.6 `incidents/{incidentId}`
```json
{
  "incidentId": "inc_20261023_001",
  "familyId": "fam_481029",
  "deviceId": "dev_win_01",
  "deviceName": "Living Room Monitor",
  "locationLabel": "Living Room (Main Area)",
  "detectedAt": "2026-10-23T21:55:00Z",
  "lastKnownPosture": "LYING",
  "transitionType": "STANDING_TO_LYING",
  "confidence": 0.92,
  "voiceResponseStatus": "NO_RESPONSE_TIMEOUT", // "CONFIRM_SAFE" | "REQUEST_HELP" | "AMBIGUOUS" | "NO_RESPONSE_TIMEOUT" | "SPEECH_ERROR"
  "voiceTranscript": "",
  "evidenceReference": "evidence/fam_481029/inc_20261023_001.jpg",
  "evidenceThumbnail": "data:image/jpeg;base64,...",
  "incidentStatus": "EMERGENCY_PUBLISHED", // "EMERGENCY_PUBLISHED" | "ACKNOWLEDGED" | "RESOLVED" | "CANCELLED_SAFE"
  "acknowledgedBy": null,
  "acknowledgedByName": null,
  "acknowledgedAt": null,
  "resolvedBy": null,
  "resolvedByName": null,
  "resolvedAt": null,
  "resolutionNote": null
}
```

### 1.7 `incidentEvents/{eventId}`
```json
{
  "eventId": "evt_991823",
  "incidentId": "inc_20261023_001",
  "actorUid": "usr_948a7b1c",
  "actorName": "Sarah Jenkins",
  "eventType": "ACKNOWLEDGED", // "SUSPECTED_FALL" | "LOCAL_ALARM_STARTED" | "VOICE_EVALUATED" | "EMERGENCY_TRIGGERED" | "ACKNOWLEDGED" | "RESOLVED"
  "timestamp": "2026-10-23T21:56:10Z",
  "metadata": {
    "note": "Checked in via living room camera"
  }
}
```
