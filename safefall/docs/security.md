# SafeFall Security, Privacy & Authorization Architecture

## 1. Principles
SafeFall adheres to strict defense-in-depth principles:
1. **Privacy-by-Design**: Normal camera frames are strictly processed in-memory and immediately destroyed. No video is ever continuously streamed or stored.
2. **Ephemeral Evidence Retention**: When an emergency is detected, an evidentiary snapshot is kept in local memory, encrypted, and uploaded to a private bucket accessible only to verified family members.
3. **No Embedded Service Account Secrets**: Desktop and mobile applications never contain Firebase Admin service-account keys or database master keys. All privileged tasks are handled via Cloud Functions or authenticated security rules.
4. **Role-Based Access Control (RBAC)**:
   - **Administrator**: Full administrative control over family settings, member invitation, role assignment, and device pairing.
   - **Caregiver**: Receives alerts, views evidence, acknowledges incidents, resolves emergencies.
   - **Viewer**: Read-only observation of family status.
5. **Fail-Safe Operation**: If network connectivity fails, the on-device alarm, microphone listening, and emergency escalation continue to function locally on the hardware.

## 2. Invitation & Pairing Token Security
- Invitation codes are cryptographically generated, random alphanumeric tokens with short lifespans (default 48 hours) and max usage counts.
- Redemptions are validated inside atomic database transactions on the server.
- Devices must present a valid pairing code to bind to a family.
