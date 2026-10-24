# SafeFall Automated Test Scenarios & Verification Matrix

This matrix operationalizes the 30 non-negotiable scenarios specified in SafeFall Part 18.

| # | Test Scenario | Input / Stimulus | Expected State Transition & Output |
|---|---|---|---|
| **1** | Standing Classification | Upright torso ($\theta < 30^\circ$), vertical bounding box | Output: `STANDING`, confidence $> 0.85$ |
| **2** | Sitting Classification | Semi-upright torso ($\theta \approx 30-40^\circ$), bent knees | Output: `SITTING`, confidence $> 0.80$ |
| **3** | Lying Classification | Horizontal torso ($\theta > 65^\circ$), wide envelope | Output: `LYING`, confidence $> 0.85$ |
| **4** | Low-Confidence Landmarking | Occluded body / low visibility score ($< 0.40$) | Output: `UNKNOWN`, no emergency trigger |
| **5** | Controlled Posture Changes | Normal transition (Standing $\to$ Sitting) over $>2.5$s | No alarm, status updates smoothly to `SITTING` |
| **6** | Rapid Fall Detection | Rapid Standing $\to$ Lying transition ($< 1.0$s) | Transition to `SUSPECTED_FALL`, start 3.0s timer |
| **7** | Local Alarm Activation | Monitored subject remains lying for $> 3.0$s | Transition to `LOCAL_ALARM` + `RESPONSE_PROCESSING` |
| **8** | Verbal Reassurance ("I'm fine")| Audio: "I am fine, don't worry" | Transition to `CANCELLED_OR_FALSE_ALARM`, alarm stops |
| **9** | Verbal Help Request ("Help me")| Audio: "Help me, I fell" | Immediate escalation to `EMERGENCY_TRIGGERED` |
| **10**| No Response Timeout | Silence / no verbal response for 10.0s | Escalation to `EMERGENCY_TRIGGERED` at $t=10.0$s |
| **11**| Speech Recognition Failure | Microphone disconnected or engine error | Fail-safe policy triggers `EMERGENCY_TRIGGERED` |
| **12**| Duplicate Fall Prevention | Rapid repeated detection within 30s window | Suppressed: Only 1 active incident per device |
| **13**| Offline Evidence Resiliency | Cloud upload fails / Network down | Incident stored locally, local alarm unaffected |
| **14**| Local Alarm Internet Independence| Fall detected while completely offline | Local alarm sounds, 10s voice check operates locally |
| **15**| Store & Forward Synchronization| Network reconnects after offline fall | Queued incident synchronizes to cloud backend |
| **16**| Family Notification Delivery | Incident reaches `EMERGENCY_PUBLISHED` | Push notification dispatched to authorized family members |
| **17**| Unauthorized Evidence Access | Request incident evidence without family membership | HTTP 403 Forbidden / Firestore rules reject access |
| **18**| Acknowledgement Semantics | Family member taps "Acknowledge" | State becomes `ACKNOWLEDGED`, alarm does not stop |
| **19**| Resolution Audit Trail | Caregiver taps "Resolve" with note | State becomes `RESOLVED`, actorUid & timestamp logged |
| **20**| Invitation Code Security | Use expired code or code past usage limit | HTTP 400/403: Invitation rejected |
| **21**| Privilege Escalation Prevention| Viewer attempts to grant themselves Admin | Firestore rules reject update with permission-denied |
| **22**| Device Authorization Check | Device submits incident with invalid family ID | Request rejected; device must be linked to valid family |
| **23**| Windows Monitor + Family App | Windows Monitor triggers fall | Windows Family App displays live emergency alert banner |
| **24**| Android Monitor + Family App | Android Monitor triggers fall | Android Family App alerts caregiver with push alert |
| **25**| iOS Monitor + Family App | iOS Monitor triggers fall | iOS Family App alerts caregiver with push alert |
| **26**| Multiplatform Consistency | Incident updated on Android | Instant real-time reflection on Windows and iOS |
| **27**| Permission Denial Gracefulness | User denies Camera or Mic permissions | App informs user clearly without crashing |
| **28**| Camera Disconnect & Recovery | USB webcam unplugged / camera lost | Monitor switches to `CAMERA_LOST`, auto-reconnects |
| **29**| Delivery Failure Accounting | Push notification service error | Delivery failure logged; retry queue updated |
| **30**| Responsive UI Adaptation | Resize desktop window / view on mobile | UI dynamically adjusts between split-pane & mobile view |
