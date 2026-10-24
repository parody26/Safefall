"""
SafeFall Dynamic Activity Classification & Fall Discrimination Test Suite.
Tests the complete matrix of daily living activities (ADLs) vs. accidental falls:
- Intentional lying down
- Sleeping & prolonged lying
- Sudden sitting down
- Bending / picking up objects
- Controlled lowering to the floor
- Simulated accidental falls (standing collapse & slumping fall)
- Partial occlusion
- Low-light degraded visibility
- Camera placement calibration presets
"""

import unittest
from src.pose.landmarks import Point3D, GeometricFeatures, extract_geometric_features
from src.pose.posture_classifier import Posture
from src.detection.temporal_features import KinematicFeatureBuffer, TemporalKinematicMetrics
from src.detection.activity_classifier import (
    ActivityState,
    CameraCalibration,
    DynamicActivityClassifier,
)
from src.detection.temporal_detector import TemporalFallDetector


class TestActivityFallDetectionMatrix(unittest.TestCase):
    def setUp(self):
        self.detector = TemporalFallDetector(
            history_window_sec=4.0,
            max_transition_duration_sec=2.2,
            min_lying_confirmation_sec=0.3,
            cooldown_sec=10.0,
        )

    def _create_synthetic_features(
        self,
        torso_angle: float,
        aspect_ratio: float,
        hip_height: float,
        knee_angle: float = 170.0,
        confidence: float = 0.95,
    ) -> GeometricFeatures:
        import math
        landmarks = [Point3D(0.5, hip_height, 0.0, confidence) for _ in range(33)]
        rad = math.radians(torso_angle)
        
        # Smooth torso tilt around hips (landmarks 11/12: shoulders, 23/24: hips)
        sh_x = 0.5 + 0.25 * math.sin(rad)
        sh_y = hip_height - 0.25 * math.cos(rad)
        landmarks[11] = Point3D(sh_x - 0.05, sh_y, 0.0, confidence)
        landmarks[12] = Point3D(sh_x + 0.05, sh_y, 0.0, confidence)
        landmarks[23] = Point3D(0.45, hip_height, 0.0, confidence)
        landmarks[24] = Point3D(0.55, hip_height, 0.0, confidence)

        # Legs extend downward/backward
        leg_rad = math.radians(max(0.0, 90.0 - torso_angle))
        kn_y = hip_height + 0.20 * math.sin(leg_rad)
        kn_x = 0.5 - 0.20 * math.cos(leg_rad)
        ak_y = kn_y + 0.18 * math.sin(leg_rad)
        ak_x = kn_x - 0.18 * math.cos(leg_rad)
        
        landmarks[25] = Point3D(kn_x - 0.03, kn_y, 0.0, confidence)
        landmarks[26] = Point3D(kn_x + 0.03, kn_y, 0.0, confidence)
        landmarks[27] = Point3D(ak_x - 0.03, ak_y, 0.0, confidence)
        landmarks[28] = Point3D(ak_x + 0.03, ak_y, 0.0, confidence)

        return extract_geometric_features(landmarks)

    def test_intentional_lying_down(self):
        """
        Matrix 1: Intentional Lying Down (Getting into bed / sofa)
        Controlled slow transition over 2.5s with low downward velocity.
        Must classify as INTENTIONAL_LYING_DOWN and NEVER trigger a fall alarm.
        """
        # Baseline standing
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=0.0)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=0.5)

        # Gradual lowering over 2.5s
        for i in range(1, 10):
            t = 0.5 + i * 0.25  # up to t=2.75
            prog = i / 10.0
            angle = 8.0 + prog * 68.0
            hip_y = 0.48 + prog * 0.38
            ar = 0.55 + prog * 0.85
            post = Posture.SITTING if prog < 0.6 else Posture.LYING
            f = self._create_synthetic_features(angle, ar, hip_y)
            event = self.detector.add_observation(post, 0.92, f, timestamp=t)
            self.assertIsNone(event, f"Intentional descent at step {i} must not trigger fall")

        # Confirmed lying on bed
        f_lying = self._create_synthetic_features(torso_angle=76.0, aspect_ratio=1.40, hip_height=0.86)
        ev_final = self.detector.add_observation(Posture.LYING, 0.94, f_lying, timestamp=3.2)
        self.assertIsNone(ev_final, "Final lying position on bed must not trigger fall")
        self.assertIn(
            self.detector.last_activity_result.activity_state,
            (ActivityState.INTENTIONAL_LYING_DOWN, ActivityState.LYING_OR_SLEEPING),
        )
        self.assertFalse(self.detector.last_activity_result.is_fall_candidate)

    def test_sleeping_and_prolonged_lying(self):
        """
        Matrix 2: Sleeping & Prolonged Lying Down
        A person resting or sleeping in a lying posture for extended periods.
        Safeguard: Prolonged lying posture must NOT independently count as a fall.
        """
        f_lying = self._create_synthetic_features(torso_angle=78.0, aspect_ratio=1.45, hip_height=0.86)

        # Person has already been lying down for 5.0 seconds
        for i in range(20):
            t = 100.0 + i * 0.25
            ev = self.detector.add_observation(Posture.LYING, 0.93, f_lying, timestamp=t)
            self.assertIsNone(ev, "Continuous lying posture must not trigger fall")

        self.assertEqual(
            self.detector.last_activity_result.activity_state,
            ActivityState.LYING_OR_SLEEPING,
        )
        self.assertFalse(self.detector.last_activity_result.is_fall_candidate)

    def test_sudden_sitting(self):
        """
        Matrix 3: Sudden Sitting Down into Chair
        Quick downward displacement, but torso angle stays upright (<45 deg)
        and ending posture is SITTING.
        Must classify as SITTING_DOWN and NOT trigger a fall alarm.
        """
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.0)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.2)

        # Rapidly sits down into an armchair in 0.4s
        f_sit = self._create_synthetic_features(torso_angle=18.0, aspect_ratio=0.90, hip_height=0.68)
        self.detector.add_observation(Posture.SITTING, 0.90, f_sit, timestamp=10.4)
        ev = self.detector.add_observation(Posture.SITTING, 0.92, f_sit, timestamp=10.6)

        self.assertIsNone(ev, "Rapid sitting down must not trigger fall event")
        self.assertEqual(
            self.detector.last_activity_result.activity_state,
            ActivityState.SITTING_DOWN,
        )

    def test_bending_and_picking_up_object(self):
        """
        Matrix 4: Bending Over & Picking Up Objects
        Person bends at the waist (torso angle reaches 60 deg), but returns upright.
        Never enters confirmed LYING posture.
        Must classify as NORMAL_MOVEMENT and NOT trigger a fall alarm.
        """
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.0)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.2)

        # Bends torso forward to reach object on floor
        f_bend = self._create_synthetic_features(torso_angle=58.0, aspect_ratio=0.85, hip_height=0.52)
        ev1 = self.detector.add_observation(Posture.STANDING, 0.85, f_bend, timestamp=10.5)
        self.assertIsNone(ev1)

        # Returns upright
        ev2 = self.detector.add_observation(Posture.STANDING, 0.92, f_stand, timestamp=11.0)
        self.assertIsNone(ev2)
        self.assertEqual(
            self.detector.last_activity_result.activity_state,
            ActivityState.NORMAL_MOVEMENT,
        )

    def test_controlled_lowering_to_floor(self):
        """
        Matrix 5: Controlled Lowering to the Floor (Yoga, stretching, exercising)
        Descent is gradual with high trajectory smoothness and low peak velocity.
        Must classify as INTENTIONAL_LYING_DOWN and NOT trigger a fall.
        """
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.95, f_stand, timestamp=20.0)

        # Controlled gentle kneeling and lowering over 3.0s
        for i in range(1, 12):
            t = 20.0 + i * 0.25
            prog = i / 12.0
            angle = 8.0 + prog * 68.0
            hip_y = 0.48 + prog * 0.38
            ar = 0.55 + prog * 0.85
            post = Posture.STANDING if prog < 0.3 else (Posture.SITTING if prog < 0.7 else Posture.LYING)
            f = self._create_synthetic_features(angle, ar, hip_y)
            ev = self.detector.add_observation(post, 0.90, f, timestamp=t)
            self.assertIsNone(ev, "Controlled lowering to floor must not trigger fall")

        self.assertIn(
            self.detector.last_activity_result.activity_state,
            (ActivityState.INTENTIONAL_LYING_DOWN, ActivityState.LYING_OR_SLEEPING),
        )
        self.assertFalse(self.detector.last_activity_result.is_fall_candidate)

    def test_simulated_accidental_falls(self):
        """
        Matrix 6: Simulated Accidental Falls
        Case A: Standing-to-floor violent collapse in 0.4s.
        Case B: Slump/slide fall from chair to floor.
        Both MUST trigger candidate fall events with high confidence.
        """
        # Case A: Rapid collapse from standing
        self.detector.reset()
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.0)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.2)

        # Rapid downward fall in 0.4s
        f_falling = self._create_synthetic_features(torso_angle=48.0, aspect_ratio=0.95, hip_height=0.72)
        self.detector.add_observation(Posture.UNKNOWN, 0.70, f_falling, timestamp=10.4)

        f_impact = self._create_synthetic_features(torso_angle=78.0, aspect_ratio=1.45, hip_height=0.86)
        ev1 = self.detector.add_observation(Posture.LYING, 0.95, f_impact, timestamp=10.6)
        self.assertIsNone(ev1, "Requires min_lying_confirmation_sec (0.3s) before triggering")

        # Confirmed lying for 0.4s -> Triggers SUSPECTED_FALL
        ev2 = self.detector.add_observation(Posture.LYING, 0.95, f_impact, timestamp=11.0)
        self.assertIsNotNone(ev2, "Accidental fall must be detected!")
        self.assertEqual(ev2.activity_state, "SUSPECTED_FALL")
        self.assertEqual(ev2.transition_type, "STANDING_TO_LYING")
        self.assertGreater(ev2.confidence, 0.80)

        # Case B: Slump fall from chair
        self.detector.reset()
        f_sit = self._create_synthetic_features(torso_angle=18.0, aspect_ratio=0.90, hip_height=0.65)
        self.detector.add_observation(Posture.SITTING, 0.92, f_sit, timestamp=50.0)
        self.detector.add_observation(Posture.SITTING, 0.92, f_sit, timestamp=50.3)

        # Slumps out of chair onto floor in 0.4s
        f_slump = self._create_synthetic_features(torso_angle=76.0, aspect_ratio=1.40, hip_height=0.86)
        self.detector.add_observation(Posture.LYING, 0.92, f_slump, timestamp=50.7)
        ev_slump = self.detector.add_observation(Posture.LYING, 0.94, f_slump, timestamp=51.1)

        self.assertIsNotNone(ev_slump, "Slump fall from sitting must trigger fall!")
        self.assertEqual(ev_slump.activity_state, "SUSPECTED_FALL")
        self.assertEqual(ev_slump.transition_type, "SITTING_TO_LYING")

    def test_partial_occlusion(self):
        """
        Matrix 7: Partial Occlusion
        Subject is partially behind furniture, causing mean visibility confidence < 0.35.
        Must classify as UNCERTAIN and NOT trigger a fall alarm.
        """
        f_stand = self._create_synthetic_features(torso_angle=8.0, aspect_ratio=0.55, hip_height=0.48)
        self.detector.add_observation(Posture.STANDING, 0.94, f_stand, timestamp=10.0)

        # Low-confidence occluded observation
        f_occluded = self._create_synthetic_features(
            torso_angle=75.0, aspect_ratio=1.40, hip_height=0.85, confidence=0.25
        )
        ev = self.detector.add_observation(Posture.LYING, 0.25, f_occluded, timestamp=10.5)

        self.assertIsNone(ev, "Occluded low-confidence tracking must not trigger fall")
        self.assertEqual(
            self.detector.last_activity_result.activity_state,
            ActivityState.UNCERTAIN,
        )

    def test_low_light_conditions(self):
        """
        Matrix 8: Low-Light Degraded Visibility Conditions
        Under low light, keypoints become noisy and confidence degrades.
        System must safely produce UNCERTAIN / NORMAL_MOVEMENT without false alarms.
        """
        f_dim = self._create_synthetic_features(
            torso_angle=12.0, aspect_ratio=0.60, hip_height=0.50, confidence=0.30
        )
        for i in range(5):
            t = 20.0 + i * 0.2
            ev = self.detector.add_observation(Posture.UNKNOWN, 0.30, f_dim, timestamp=t)
            self.assertIsNone(ev, "Low-light degraded tracking must not produce false fall")

        self.assertEqual(
            self.detector.last_activity_result.activity_state,
            ActivityState.UNCERTAIN,
        )

    def test_camera_calibration_presets(self):
        """
        Matrix 9: Camera Placement Presets
        Verifies calibration presets for table_level, eye_level, and ceiling_mount.
        """
        table_calib = CameraCalibration.for_placement("table_level")
        eye_calib = CameraCalibration.for_placement("eye_level")
        ceiling_calib = CameraCalibration.for_placement("ceiling_mount")

        self.assertEqual(table_calib.camera_placement, "table_level")
        self.assertEqual(eye_calib.camera_placement, "eye_level")
        self.assertEqual(ceiling_calib.camera_placement, "ceiling_mount")

        # Ceiling mount should have lower vertical velocity threshold due to foreshortening
        self.assertLess(ceiling_calib.fall_velocity_threshold, table_calib.fall_velocity_threshold)
        self.assertLess(ceiling_calib.fall_velocity_threshold, eye_calib.fall_velocity_threshold)


if __name__ == "__main__":
    unittest.main()
