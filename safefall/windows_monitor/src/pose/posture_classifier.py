"""
SafeFall Posture Classification Module.
Classifies human posture into STANDING, SITTING, LYING, or UNKNOWN
using multi-feature biomechanical heuristics.
"""

from enum import Enum
from typing import Tuple, Optional
from .landmarks import GeometricFeatures


class Posture(str, Enum):
    STANDING = "STANDING"
    SITTING = "SITTING"
    LYING = "LYING"
    UNKNOWN = "UNKNOWN"


class PostureClassifier:
    def __init__(
        self,
        min_visibility_threshold: float = 0.30,
        lying_torso_angle_min: float = 48.0,
        standing_torso_angle_max: float = 35.0,
        sitting_knee_flexion_min: float = 60.0,
        sitting_knee_flexion_max: float = 140.0,
        lying_aspect_ratio_min: float = 1.05,
        standing_aspect_ratio_max: float = 0.85,
    ):
        self.min_visibility_threshold = min_visibility_threshold
        self.lying_torso_angle_min = lying_torso_angle_min
        self.standing_torso_angle_max = standing_torso_angle_max
        self.sitting_knee_flexion_min = sitting_knee_flexion_min
        self.sitting_knee_flexion_max = sitting_knee_flexion_max
        self.lying_aspect_ratio_min = lying_aspect_ratio_min
        self.standing_aspect_ratio_max = standing_aspect_ratio_max

    def classify(self, feat: Optional[GeometricFeatures]) -> Tuple[Posture, float]:
        """
        Evaluates features and returns (Posture, confidence).
        Returns UNKNOWN if landmarks are occluded, confidence < threshold, or features are None.
        """
        if feat is None:
            return Posture.UNKNOWN, 0.0

        if feat.mean_visibility < self.min_visibility_threshold:
            return Posture.UNKNOWN, feat.mean_visibility

        torso_angle = feat.torso_angle_deg
        ar = feat.aspect_ratio
        avg_knee = (feat.left_knee_flexion_deg + feat.right_knee_flexion_deg) / 2.0
        hip_y = feat.relative_hip_height

        # 1. LYING: Torso is predominantly horizontal OR bounding box is wide near floor
        if torso_angle >= self.lying_torso_angle_min or (ar >= self.lying_aspect_ratio_min and hip_y > 0.60):
            # Confidence weighted by how horizontal the torso is
            conf = min(0.99, max(0.60, torso_angle / 90.0 * 0.7 + (ar / 2.0) * 0.3))
            return Posture.LYING, round(conf, 2)

        # 2. STANDING: Torso upright, tall bounding box, extended knees
        if torso_angle <= self.standing_torso_angle_max and ar <= self.standing_aspect_ratio_max:
            if avg_knee > 140.0:
                conf = min(0.98, max(0.75, (1.0 - torso_angle / 90.0) * 0.6 + (avg_knee / 180.0) * 0.4))
                return Posture.STANDING, round(conf, 2)

        # 3. SITTING: Torso relatively upright, bent knees, intermediate aspect ratio
        if torso_angle <= 48.0 and self.sitting_knee_flexion_min <= avg_knee <= self.sitting_knee_flexion_max:
            conf = min(0.95, max(0.65, 0.5 + 0.5 * (1.0 - abs(avg_knee - 95.0) / 45.0)))
            return Posture.SITTING, round(conf, 2)

        # 4. Secondary evaluation for standing vs sitting based on aspect ratio
        if ar < 0.85 and torso_angle < 40.0:
            return Posture.STANDING, 0.70

        if 0.80 <= ar <= 1.25 and torso_angle < 55.0:
            return Posture.SITTING, 0.68

        # Fallback for ambiguous orientations
        return Posture.UNKNOWN, 0.50
