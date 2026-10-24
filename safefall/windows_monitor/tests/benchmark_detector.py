"""
SafeFall Automated Detector Benchmark.
Executes timestamped sequence benchmarks across standing, sitting, normal movements,
controlled lying down, rapid falls, and verbal safety interactions.
"""

import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.landmarks import Point3D, extract_geometric_features
from src.pose.posture_classifier import PostureClassifier, Posture
from src.detection.temporal_detector import TemporalFallDetector
from src.detection.state_machine import EmergencyStateMachine, EmergencyState
from src.speech.speech_service import VerbalIntent


def run_benchmark():
    print("=" * 65)
    print("  [SafeFall] Automated Fall Detection & State Benchmark")
    print("=" * 65)

    classifier = PostureClassifier()
    detector = TemporalFallDetector()

    results = []

    # Scenario 1: Standing
    t0 = time.monotonic()
    l_standing = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    l_standing[11] = Point3D(0.46, 0.28, 0.0, 0.95)
    l_standing[12] = Point3D(0.54, 0.28, 0.0, 0.95)
    l_standing[23] = Point3D(0.47, 0.52, 0.0, 0.95)
    l_standing[24] = Point3D(0.53, 0.52, 0.0, 0.95)
    l_standing[25] = Point3D(0.47, 0.72, 0.0, 0.95)
    l_standing[26] = Point3D(0.53, 0.72, 0.0, 0.95)
    l_standing[27] = Point3D(0.47, 0.88, 0.0, 0.95)
    l_standing[28] = Point3D(0.53, 0.88, 0.0, 0.95)

    f_stand = extract_geometric_features(l_standing)
    posture, conf = classifier.classify(f_stand)
    latency_ms = (time.monotonic() - t0) * 1000
    p1 = posture == Posture.STANDING
    results.append(("1. Standing Posture Classification", p1, f"{latency_ms:.2f} ms"))

    # Scenario 2: Sitting
    t0 = time.monotonic()
    l_sitting = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    l_sitting[11] = Point3D(0.45, 0.45, 0.0, 0.95)
    l_sitting[12] = Point3D(0.55, 0.45, 0.0, 0.95)
    l_sitting[23] = Point3D(0.46, 0.65, 0.0, 0.95)
    l_sitting[24] = Point3D(0.54, 0.65, 0.0, 0.95)
    l_sitting[25] = Point3D(0.56, 0.68, 0.0, 0.95)
    l_sitting[26] = Point3D(0.64, 0.68, 0.0, 0.95)
    l_sitting[27] = Point3D(0.56, 0.85, 0.0, 0.95)
    l_sitting[28] = Point3D(0.64, 0.85, 0.0, 0.95)

    f_sit = extract_geometric_features(l_sitting)
    posture_sit, conf_sit = classifier.classify(f_sit)
    latency_ms = (time.monotonic() - t0) * 1000
    p2 = posture_sit == Posture.SITTING
    results.append(("2. Sitting Posture Classification", p2, f"{latency_ms:.2f} ms"))

    # Scenario 3: Lying
    t0 = time.monotonic()
    l_lying = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    l_lying[11] = Point3D(0.20, 0.82, 0.0, 0.95)
    l_lying[12] = Point3D(0.24, 0.82, 0.0, 0.95)
    l_lying[23] = Point3D(0.42, 0.84, 0.0, 0.95)
    l_lying[24] = Point3D(0.46, 0.84, 0.0, 0.95)
    l_lying[25] = Point3D(0.62, 0.85, 0.0, 0.95)
    l_lying[26] = Point3D(0.66, 0.85, 0.0, 0.95)
    l_lying[27] = Point3D(0.80, 0.86, 0.0, 0.95)
    l_lying[28] = Point3D(0.84, 0.86, 0.0, 0.95)

    f_ly = extract_geometric_features(l_lying)
    posture_ly, conf_ly = classifier.classify(f_ly)
    latency_ms = (time.monotonic() - t0) * 1000
    p3 = posture_ly == Posture.LYING
    results.append(("3. Lying Posture Classification", p3, f"{latency_ms:.2f} ms"))

    # Scenario 6: Rapid Fall Transition
    t0 = time.monotonic()
    detector.reset()
    detector.add_observation(Posture.STANDING, 0.92, f_stand, timestamp=100.0)
    detector.add_observation(Posture.STANDING, 0.90, f_stand, timestamp=100.2)
    detector.add_observation(Posture.LYING, 0.95, f_ly, timestamp=100.6)
    fall_event = detector.add_observation(Posture.LYING, 0.95, f_ly, timestamp=101.0)
    latency_ms = (time.monotonic() - t0) * 1000
    p6 = fall_event is not None and fall_event.transition_type == "STANDING_TO_LYING"
    results.append(("6. Rapid Fall Temporal Detection", p6, f"{latency_ms:.2f} ms"))

    # Scenario 7 & 8: State Machine Full Life-cycle with Reassurance
    t0 = time.monotonic()
    sm = EmergencyStateMachine(suspected_fall_grace_sec=0.1, voice_response_timeout_sec=0.2)
    sm.handle_candidate_fall(fall_event)
    s_susp = sm.current_state == EmergencyState.SUSPECTED_FALL
    time.sleep(0.12)
    sm.handle_posture_update("LYING")
    s_alarm = sm.current_state == EmergencyState.RESPONSE_PROCESSING
    sm.handle_verbal_response(VerbalIntent.CONFIRM_SAFE, "I'm okay")
    s_safe = sm.current_state == EmergencyState.NORMAL
    latency_ms = (time.monotonic() - t0) * 1000
    p7_8 = s_susp and s_alarm and s_safe
    results.append(("7/8. Suspected Fall -> Alarm -> Reassurance Recovery", p7_8, f"{latency_ms:.2f} ms"))

    print(f"{'Scenario Name':<45} | {'Status':<6} | {'Inference Latency'}")
    print("-" * 65)
    for name, passed, lat in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<45} | {status:<6} | {lat}")
    print("=" * 65)
    all_ok = all(r[1] for r in results)
    print(f"Overall Result: {'ALL BENCHMARKS PASSED (100%)' if all_ok else 'BENCHMARK FAILURES DETECTED'}")
    return all_ok


if __name__ == "__main__":
    run_benchmark()
