# SafeFall Automated Testing & Verification Guide

## 1. Test Architecture
SafeFall provides automated test coverage across:
1. **Mathematical Feature Extraction**: Torso angles, knee flexion, aspect ratio calculation, hip velocity.
2. **Temporal Fall Detector**: Distinguishing controlled transitions vs rapid falls.
3. **Emergency State Machine**: Monotonic 3-second grace countdown, 10-second response window, intent resolution, escalation.
4. **Verbal Intent Classifier**: Prioritizing `REQUEST_HELP` over reassurance, handling silence and ambiguity.
5. **Role-Based Security & Permissions**: Testing unauthorized incident access and invitation abuse.

## 2. Running Automated Tests

### 2.1 Windows Monitor & Computer Vision Tests
```powershell
cd c:\Users\katar\OneDrive\Desktop\SYNTAX\safefall\windows_monitor
python -m unittest discover tests -v
```

### 2.2 Flutter Application Tests
```powershell
cd c:\Users\katar\OneDrive\Desktop\SYNTAX\safefall\flutter_apps
flutter test
```

### 2.3 Synthetic Posture Simulation Benchmark
```powershell
cd c:\Users\katar\OneDrive\Desktop\SYNTAX\safefall\windows_monitor
python tests/benchmark_detector.py
```
Outputs precision, recall, latency, and state machine transition traces across all 30 standard benchmark scenarios.
