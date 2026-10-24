"""
Unit Tests for SafeFall Temporal Fall Detector.
"""

import unittest
from src.pose.landmarks import Point3D, extract_geometric_features
from src.pose.posture_classifier import Posture
from src.detection.temporal_detector import TemporalFallDetector


class TestTemporalDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TemporalFallDetector(
            history_window_sec=4.0,
            max_transition_duration_sec=1.4,
            min_lying_confirmation_sec=0.3,
            cooldown_sec=10.0,
        )

    def _make_features(self, torso_angle: float, aspect_ratio: float, hip_height: float):
        landmarks = [Point3D(0.5, 0.5, 0.0, 0.95) for _ in range(33)]
        # Customize landmarks for desired geometry
        if torso_angle > 50:
            landmarks[11] = Point3D(0.2, hip_height, 0.0, 0.95)
            landmarks[23] = Point3D(0.4, hip_height + 0.02, 0.0, 0.95)
            landmarks[25] = Point3D(0.6, hip_height + 0.03, 0.0, 0.95)
            landmarks[27] = Point3D(0.8, hip_height + 0.04, 0.0, 0.95)
        else:
            landmarks[11] = Point3D(0.5, hip_height - 0.25, 0.0, 0.95)
            landmarks[23] = Point3D(0.5, hip_height, 0.0, 0.95)
            landmarks[25] = Point3D(0.5, hip_height + 0.20, 0.0, 0.95)
            landmarks[27] = Point3D(0.5, hip_height + 0.38, 0.0, 0.95)
        return extract_geometric_features(landmarks)

    def test_rapid_fall_detection(self):
        """Scenario 6: Rapid Standing -> Lying transition in 0.6s triggers fall event."""
        # t=10.0: Standing
        f_stand = self._make_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.50)
        self.detector.add_observation(Posture.STANDING, 0.92, f_stand, timestamp=10.0)

        # t=10.2: Still Standing
        self.detector.add_observation(Posture.STANDING, 0.90, f_stand, timestamp=10.2)

        # t=10.4: Intermediate descent
        f_descending = self._make_features(torso_angle=45.0, aspect_ratio=0.85, hip_height=0.70)
        self.detector.add_observation(Posture.UNKNOWN, 0.60, f_descending, timestamp=10.4)

        # t=10.6: Lying down on floor
        f_lying = self._make_features(torso_angle=75.0, aspect_ratio=1.40, hip_height=0.85)
        event1 = self.detector.add_observation(Posture.LYING, 0.94, f_lying, timestamp=10.6)
        # Needs min_lying_confirmation_sec (0.3s)
        self.assertIsNone(event1)

        # t=11.0: Still lying down after 0.4s
        event2 = self.detector.add_observation(Posture.LYING, 0.95, f_lying, timestamp=11.0)
        self.assertIsNotNone(event2)
        self.assertEqual(event2.transition_type, "STANDING_TO_LYING")
        self.assertGreater(event2.confidence, 0.80)

    def test_slow_controlled_movement_not_fall(self):
        """Scenario 5: Slow transition (e.g. slowly lying down on bed over 3.0s) is NOT a fall."""
        f_stand = self._make_features(8.0, 0.55, 0.50)
        self.detector.add_observation(Posture.STANDING, 0.90, f_stand, timestamp=10.0)

        # 3.0 seconds elapsed before reaching lying position
        f_lying = self._make_features(75.0, 1.40, 0.85)
        self.detector.add_observation(Posture.LYING, 0.92, f_lying, timestamp=13.0)
        event = self.detector.add_observation(Posture.LYING, 0.92, f_lying, timestamp=13.5)
        self.assertIsNone(event, "Slow transition over 3 seconds must not trigger fall")

    def test_duplicate_prevention(self):
        """Scenario 12: Duplicate detections within cooldown window are suppressed."""
        f_stand = self._make_features(8.0, 0.55, 0.50)
        f_lying = self._make_features(75.0, 1.40, 0.85)

        self.detector.add_observation(Posture.STANDING, 0.90, f_stand, timestamp=10.0)
        self.detector.add_observation(Posture.LYING, 0.95, f_lying, timestamp=10.5)
        ev1 = self.detector.add_observation(Posture.LYING, 0.95, f_lying, timestamp=10.9)
        self.assertIsNotNone(ev1)

        # Subsequent observation within cooldown
        ev2 = self.detector.add_observation(Posture.LYING, 0.95, f_lying, timestamp=12.0)
        self.assertIsNone(ev2, "Cooldown must suppress duplicate alarms for continuous event")


if __name__ == "__main__":
    unittest.main()
