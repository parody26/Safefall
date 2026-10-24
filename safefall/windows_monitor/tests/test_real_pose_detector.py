"""
Unit Tests for Optical RealPoseDetector and Landmark Skeleton Drawing.
"""

import unittest
import numpy as np
from src.pose.real_pose_detector import RealPoseDetector


class TestRealPoseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RealPoseDetector()

    def test_detector_initialization(self):
        """Validates that YOLO pose model is loaded and available."""
        self.assertTrue(self.detector.is_available())
        self.assertTrue(self.detector.show_landmarks)

    def test_detect_landmarks_on_blank_frame(self):
        """A blank frame should not crash and should return None for landmarks."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, annotated_frame = self.detector.detect_landmarks(frame, draw_overlay=True)
        self.assertIsNone(landmarks)
        self.assertEqual(annotated_frame.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
