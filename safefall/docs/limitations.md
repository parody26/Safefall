# SafeFall Platform Limitations & Engineering Constraints

Per project requirements, this document transparently states known hardware, OS, and environmental constraints.

## 1. Computer Vision & Landmark Limitations
1. **Periodic Frame Sampling**: SafeFall samples frames periodically (default 5–10 Hz) to maintain low CPU/power consumption. Periodic snapshots cannot calculate instantaneous kinetic impact forces; fall detection relies on temporal displacement, velocity, and orientation changes across consecutive observations.
2. **Heavy Occlusion**: If a subject is underneath thick blankets, behind furniture, or completely occluded, landmark confidence drops below 0.40, categorizing the frame as `UNKNOWN`. A fall hidden behind a bed frame cannot be inferred solely by optical camera.
3. **Multi-Person Scenes**: In scenes with $>1$ individual, identity crossovers can occur without high-overhead 3D tracking. SafeFall flags ambiguous scenes and suspends automated emergency escalation to prevent false alarms on family visitors.
4. **Lighting Conditions**: Standard optical webcams degrade in complete darkness. Infra-red (IR) night-vision or low-light sensors are recommended for dark bedrooms.

## 2. Mobile OS Background Execution Constraints
1. **iOS Background Camera Policy**: iOS strictly terminates camera capture when an application transitions to the background or the screen locks. The iPhone Monitoring Application must run in **foreground kiosk mode** (with screen brightness dimmed and wakelock enabled) when actively monitoring.
2. **Android Battery Optimization**: Aggressive manufacturer battery savers (e.g. Samsung OneUI, Xiaomi MIUI) may kill background processes. The Android Monitor requires Foreground Service notification and battery optimization exemption.

## 3. Speech Recognition Constraints
1. **Acoustic Feedback**: While the local repeating beep-beep alarm is sounding, speech-to-text algorithms must handle acoustic echo. SafeFall interleaves alarm beeps with listening pauses or applies software echo suppression.
2. **Speech Engine Latency**: Cloud-based STT requires active Internet. On-device engines (e.g. Windows Speech API, Android SpeechRecognizer, iOS SFSpeechRecognizer) are preferred for local resilience.

## 4. Emergency Calling
1. SafeFall prepares and prompts the device's native telephone dialer with the designated regional emergency number (e.g. 112 in India / Europe, 911 in US).
2. Automated programmatic call placement without human verification is subject to carrier safety rules to prevent illegal 911 auto-dialing. SafeFall requires user tap confirmation or dedicated VoIP escalation.
