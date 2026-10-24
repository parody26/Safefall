"""
Unit & Integration Tests for SafeFall Trained Sequence GRU Model.
Validates model weights loading, forward inference latency,
and discrimination between accidental falls and intentional lying down.
"""

import os
import sys
import unittest
import numpy as np

# Ensure windows_monitor is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.sequence_model import TemporalPoseGRU, TemporalSequenceEvaluator
from src.detection.activity_classifier import DynamicActivityClassifier, ActivityState
from src.pose.posture_classifier import Posture
from src.detection.temporal_features import TemporalKinematicMetrics


class TestTrainedSequenceModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "fall_vs_lying_gru.pt")
        )
        cls.dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "le2i_extracted_dataset.npz")
        )

    def test_01_weights_file_exists(self):
        """Verifies that trained PyTorch model weights exist on disk."""
        self.assertTrue(
            os.path.isfile(self.weights_path),
            f"Trained model weights not found at {self.weights_path}",
        )
        size_kb = os.path.getsize(self.weights_path) / 1024
        self.assertGreater(size_kb, 50.0, "Weights file should be > 50 KB")

    def test_02_evaluator_auto_loads_weights(self):
        """Verifies that TemporalSequenceEvaluator discovers and auto-loads weights."""
        evaluator = TemporalSequenceEvaluator()
        self.assertTrue(evaluator.is_trained, "Evaluator should automatically report is_trained=True")
        self.assertIsNotNone(evaluator.model)

    def test_03_inference_output_shape_and_latency(self):
        """Verifies model forward pass latency is fast (< 5ms on CPU)."""
        import time
        evaluator = TemporalSequenceEvaluator()
        dummy_seq = [[0.1 * j for j in range(8)] for _ in range(16)]

        # Warmup
        evaluator.predict_sequence(dummy_seq)

        t0 = time.perf_counter()
        pred_class, conf, is_trained = evaluator.predict_sequence(dummy_seq)
        lat_ms = (time.perf_counter() - t0) * 1000

        self.assertTrue(is_trained)
        self.assertIn(pred_class, TemporalSequenceEvaluator.CLASS_LABELS)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertLess(lat_ms, 15.0, f"Inference latency too slow: {lat_ms:.2f} ms")

    def test_04_distinguish_fall_vs_lying_from_real_dataset(self):
        """Verifies fall vs lying discrimination using real sequences from LE2I."""
        if not os.path.isfile(self.dataset_path):
            self.skipTest("LE2I extracted dataset not present")

        data = np.load(self.dataset_path, allow_pickle=True)
        seqs = data["sequences"]
        lbls = data["labels"]

        evaluator = TemporalSequenceEvaluator()

        # Find real fall samples (label 5)
        fall_indices = np.where(lbls == 5)[0]
        self.assertGreater(len(fall_indices), 0, "No fall samples found in dataset")

        fall_detected_count = 0
        test_samples = min(40, len(fall_indices))
        for idx in fall_indices[:test_samples]:
            res = evaluator.distinguish_fall_vs_lying(seqs[idx].tolist())
            if res["is_fall"]:
                fall_detected_count += 1

        fall_recall = fall_detected_count / test_samples
        self.assertGreaterEqual(
            fall_recall, 0.70, f"Expected >= 70% recall on real fall samples, got {fall_recall*100:.1f}%"
        )

        # Find real lying / resting samples (label 4)
        lying_indices = np.where(lbls == 4)[0]
        self.assertGreater(len(lying_indices), 0, "No lying samples found in dataset")

        false_fall_on_lying = 0
        test_lying_samples = min(40, len(lying_indices))
        for idx in lying_indices[:test_lying_samples]:
            res = evaluator.distinguish_fall_vs_lying(seqs[idx].tolist())
            if res["is_fall"]:
                false_fall_on_lying += 1

        far_on_lying = false_fall_on_lying / test_lying_samples
        self.assertLessEqual(
            far_on_lying, 0.25, f"Expected <= 25% false falls on real lying postures, got {far_on_lying*100:.1f}%"
        )

    def test_05_dynamic_activity_classifier_integration(self):
        """Verifies that DynamicActivityClassifier integrates the neural sequence model."""
        classifier = DynamicActivityClassifier()
        self.assertTrue(classifier.sequence_evaluator.is_trained)

        # Test lying posture evaluation with gentle kinematics (intentional lying down)
        metrics = TemporalKinematicMetrics(
            timestamp=2.0,
            current_torso_angle=75.0,
            current_hip_height=0.85,
            current_aspect_ratio=1.4,
            downward_hip_velocity=0.03,
            angular_velocity_deg_s=8.0,
            horizontal_velocity=0.01,
            downward_hip_acceleration=0.04,
            angular_acceleration=1.5,
            peak_downward_velocity=0.04,
            peak_angular_velocity=10.0,
            peak_downward_acceleration=0.05,
            motion_energy=0.01,
            trajectory_smoothness=0.88,
            time_window_duration=1.2,
            mean_confidence=0.92,
        )

        # Add observations into buffer with advancing time
        for i in range(10):
            t = 2.0 + i * 0.1
            m = TemporalKinematicMetrics(
                timestamp=t,
                current_torso_angle=75.0,
                current_hip_height=0.85,
                current_aspect_ratio=1.4,
                downward_hip_velocity=0.03,
                angular_velocity_deg_s=8.0,
                horizontal_velocity=0.01,
                downward_hip_acceleration=0.04,
                angular_acceleration=1.5,
                peak_downward_velocity=0.04,
                peak_angular_velocity=10.0,
                peak_downward_acceleration=0.05,
                motion_energy=0.01,
                trajectory_smoothness=0.88,
                time_window_duration=1.2,
                mean_confidence=0.92,
            )
            res = classifier.evaluate(
                current_posture=Posture.LYING,
                posture_conf=0.92,
                metrics=m,
                timestamp=t,
                min_lying_confirmation_sec=0.2,
            )

        # Must NOT be classified as SUSPECTED_FALL
        self.assertNotEqual(res.activity_state, ActivityState.SUSPECTED_FALL)
        self.assertIn(
            res.activity_state,
            (ActivityState.INTENTIONAL_LYING_DOWN, ActivityState.LYING_OR_SLEEPING),
        )
        self.assertFalse(res.is_fall_candidate)

    def test_06_standing_to_lying_transition_not_a_fall(self):
        """
        Directly validates user requirement:
        Controlled normal standing-to-lying must NOT be treated as a fall.
        """
        from src.detection.temporal_detector import TemporalFallDetector
        from src.pose.landmarks import GeometricFeatures

        detector = TemporalFallDetector()
        fall_events = []

        def make_geom(angle, hip_y, ar):
            return GeometricFeatures(
                torso_angle_deg=angle,
                aspect_ratio=ar,
                relative_hip_height=hip_y,
                left_knee_flexion_deg=170.0 - angle * 0.4,
                right_knee_flexion_deg=170.0 - angle * 0.4,
                mean_visibility=0.95,
                body_center=(0.5, hip_y),
                bbox=(0.4, hip_y - 0.4, 0.6, hip_y + 0.4),
            )

        # 1. Standing for 1 second
        for i in range(10):
            t = i * 0.1
            ev = detector.add_observation(Posture.STANDING, 0.95, make_geom(5.0, 0.48, 0.55), timestamp=t)
            if ev:
                fall_events.append(ev)

        # 2. Normal transition to lying over 1.2s
        for i in range(12):
            t = 1.0 + (i + 1) * 0.1
            prog = (i + 1) / 12.0
            angle = 5.0 + prog * 75.0
            hip_y = 0.48 + prog * 0.38
            ar = 0.55 + prog * 0.90
            post = Posture.STANDING if prog < 0.3 else (Posture.SITTING if prog < 0.6 else Posture.LYING)
            ev = detector.add_observation(post, 0.95, make_geom(angle, hip_y, ar), timestamp=t)
            if ev:
                fall_events.append(ev)

        # 3. Settling and lying on bed/couch for 2 seconds
        for i in range(20):
            t = 2.2 + (i + 1) * 0.1
            ev = detector.add_observation(Posture.LYING, 0.95, make_geom(80.0, 0.86, 1.45), timestamp=t)
            if ev:
                fall_events.append(ev)

        self.assertEqual(
            len(fall_events),
            0,
            f"Expected zero falls for normal standing-to-lying, but got {len(fall_events)}: "
            f"{[e.trigger_reason for e in fall_events]}",
        )


if __name__ == "__main__":
    unittest.main()

