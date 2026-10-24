"""
Unit Tests for Biomechanical Landmark Extraction & Posture Classification.
"""

import unittest
from src.pose.landmarks import Point3D, extract_geometric_features
from src.pose.posture_classifier import PostureClassifier, Posture


class TestPoseClassification(unittest.TestCase):
    def setUp(self):
        self.classifier = PostureClassifier()

    def test_standing_posture_classification(self):
        """Scenario 1: Upright standing person classified correctly."""
        landmarks = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        # Shoulders upright
        landmarks[11] = Point3D(0.46, 0.28, 0.0, 0.95)
        landmarks[12] = Point3D(0.54, 0.28, 0.0, 0.95)
        # Hips below shoulders
        landmarks[23] = Point3D(0.47, 0.52, 0.0, 0.95)
        landmarks[24] = Point3D(0.53, 0.52, 0.0, 0.95)
        # Knees extended
        landmarks[25] = Point3D(0.47, 0.72, 0.0, 0.95)
        landmarks[26] = Point3D(0.53, 0.72, 0.0, 0.95)
        # Ankles
        landmarks[27] = Point3D(0.47, 0.88, 0.0, 0.95)
        landmarks[28] = Point3D(0.53, 0.88, 0.0, 0.95)

        features = extract_geometric_features(landmarks)
        self.assertIsNotNone(features)
        self.assertLess(features.torso_angle_deg, 30.0)
        self.assertLess(features.aspect_ratio, 0.80)

        posture, conf = self.classifier.classify(features)
        self.assertEqual(posture, Posture.STANDING)
        self.assertGreaterEqual(conf, 0.75)

    def test_lying_posture_classification(self):
        """Scenario 3: Horizontal lying person classified correctly."""
        landmarks = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        # Person horizontal on floor (torso nearly parallel to horizontal axis)
        landmarks[11] = Point3D(0.20, 0.82, 0.0, 0.95)
        landmarks[12] = Point3D(0.24, 0.82, 0.0, 0.95)
        landmarks[23] = Point3D(0.42, 0.84, 0.0, 0.95)
        landmarks[24] = Point3D(0.46, 0.84, 0.0, 0.95)
        landmarks[25] = Point3D(0.62, 0.85, 0.0, 0.95)
        landmarks[26] = Point3D(0.66, 0.85, 0.0, 0.95)
        landmarks[27] = Point3D(0.80, 0.86, 0.0, 0.95)
        landmarks[28] = Point3D(0.84, 0.86, 0.0, 0.95)

        features = extract_geometric_features(landmarks)
        self.assertIsNotNone(features)
        self.assertGreater(features.torso_angle_deg, 60.0)
        self.assertGreater(features.aspect_ratio, 1.10)

        posture, conf = self.classifier.classify(features)
        self.assertEqual(posture, Posture.LYING)
        self.assertGreaterEqual(conf, 0.70)

    def test_low_confidence_unknown_posture(self):
        """Scenario 4: Low-confidence/occluded landmarks produce UNKNOWN."""
        landmarks = [Point3D(0.5, 0.5, 0.0, 0.20) for _ in range(33)] # low visibility
        features = extract_geometric_features(landmarks)
        posture, conf = self.classifier.classify(features)
        self.assertEqual(posture, Posture.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
