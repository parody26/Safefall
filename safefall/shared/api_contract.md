# SafeFall Unified API & Protocol Contract

## 1. Overview
SafeFall provides dual interoperability:
1. **Direct Cloud Firestore Client SDK bindings** for high-performance real-time reactive streams (Flutter & Python clients).
2. **REST / HTTP Edge API Gateway** for headless edge monitors, microservices, and fallback connectivity.

## 2. Authentication Protocol
- Bearer Token: Firebase JWT / Auth ID Token (`Authorization: Bearer <idToken>`).
- Claims: Contains `uid`, `email`, and optionally custom family claims.

## 3. Endpoints

### 3.1 Device Pairing
`POST /api/v1/devices/pair`
- **Request**:
```json
{
  "familyId": "fam_481029",
  "deviceName": "Living Room Monitor",
  "platform": "windows",
  "locationLabel": "Living Room",
  "pairingCode": "PAIR-9912"
}
```
- **Response (200 OK)**:
```json
{
  "deviceId": "dev_win_01",
  "apiKey": "sec_dev_token_xxx",
  "status": "paired"
}
```

### 3.2 Publish Incident
`POST /api/v1/incidents`
- **Request**:
```json
{
  "familyId": "fam_481029",
  "deviceId": "dev_win_01",
  "detectedAt": "2026-10-23T21:55:00Z",
  "lastKnownPosture": "LYING",
  "transitionType": "STANDING_TO_LYING",
  "confidence": 0.92,
  "voiceResponseStatus": "NO_RESPONSE_TIMEOUT",
  "evidenceBase64": "/9j/4AAQSkZJRg...",
  "status": "EMERGENCY_PUBLISHED"
}
```
- **Response (201 Created)**:
```json
{
  "incidentId": "inc_20261023_001",
  "status": "published",
  "notifiedMembersCount": 3
}
```

### 3.3 Acknowledge Incident
`POST /api/v1/incidents/{incidentId}/acknowledge`
- **Request**:
```json
{
  "familyId": "fam_481029",
  "actorUid": "usr_948a7b1c",
  "actorName": "Sarah Jenkins"
}
```
- **Response (200 OK)**:
```json
{
  "incidentId": "inc_20261023_001",
  "status": "ACKNOWLEDGED",
  "acknowledgedAt": "2026-10-23T21:56:10Z"
}
```

### 3.4 Resolve Incident
`POST /api/v1/incidents/{incidentId}/resolve`
- **Request**:
```json
{
  "familyId": "fam_481029",
  "actorUid": "usr_948a7b1c",
  "actorName": "Sarah Jenkins",
  "resolutionNote": "Checked Grandma, she tripped over rug but is uninjured and back in bed."
}
```
- **Response (200 OK)**:
```json
{
  "incidentId": "inc_20261023_001",
  "status": "RESOLVED",
  "resolvedAt": "2026-10-23T21:58:30Z"
}
```
