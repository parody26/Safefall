"""
SafeFall Body Landmark Processing & Geometric Feature Extraction.
Supports standard 33-point pose landmark formats (MediaPipe / COCO / YOLO).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class GeometricFeatures:
    torso_angle_deg: float          # 0 deg = vertical/upright, 90 deg = horizontal
    aspect_ratio: float             # bounding box width / height
    relative_hip_height: float      # normalized 0.0 (top) to 1.0 (bottom/floor)
    left_knee_flexion_deg: float    # angle at left knee
    right_knee_flexion_deg: float   # angle at right knee
    mean_visibility: float          # average landmark confidence
    body_center: Tuple[float, float]# (cx, cy)
    bbox: Tuple[float, float, float, float] # (min_x, min_y, max_x, max_y)


# MediaPipe landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
NOSE = 0


def calculate_angle_3points(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """Calculates angle at p2 formed by (p1 - p2) and (p3 - p2) in degrees."""
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    
    if mag1 * mag2 < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def extract_geometric_features(landmarks: List[Point3D]) -> Optional[GeometricFeatures]:
    """
    Extracts calibrated geometric and biomechanical features from 33 body landmarks.
    Returns None if essential landmarks are missing or confidence is critically low.
    """
    if len(landmarks) < 29:
        return None

    # Key landmark references
    l_sh = landmarks[L_SHOULDER]
    r_sh = landmarks[R_SHOULDER]
    l_hp = landmarks[L_HIP]
    r_hp = landmarks[R_HIP]
    l_kn = landmarks[L_KNEE]
    r_kn = landmarks[R_KNEE]
    l_ak = landmarks[L_ANKLE]
    r_ak = landmarks[R_ANKLE]

    key_landmarks = [l_sh, r_sh, l_hp, r_hp, l_kn, r_kn, l_ak, r_ak]
    mean_vis = sum(pt.visibility for pt in key_landmarks) / len(key_landmarks)

    # 1. Torso midpoint calculations
    sh_mid_x = (l_sh.x + r_sh.x) / 2.0
    sh_mid_y = (l_sh.y + r_sh.y) / 2.0

    hp_mid_x = (l_hp.x + r_hp.x) / 2.0
    hp_mid_y = (l_hp.y + r_hp.y) / 2.0

    # Torso vector from hips to shoulders
    dx = sh_mid_x - hp_mid_x
    dy = sh_mid_y - hp_mid_y

    # Torso angle relative to vertical axis (0 = perfectly upright, 90 = lying horizontal)
    # Note: in image coordinates, y increases downwards.
    torso_angle_deg = math.degrees(math.atan2(abs(dx), max(1e-6, abs(dy))))

    # 2. Bounding envelope & Aspect Ratio
    all_x = [pt.x for pt in key_landmarks if pt.visibility > 0.2]
    all_y = [pt.y for pt in key_landmarks if pt.visibility > 0.2]

    if not all_x or not all_y:
        return None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max(1e-4, max_x - min_x)
    height = max(1e-4, max_y - min_y)
    aspect_ratio = width / height

    # 3. Relative hip height in normalized frame
    relative_hip_height = hp_mid_y

    # 4. Knee flexion
    l_knee_angle = calculate_angle_3points(l_hp, l_kn, l_ak)
    r_knee_angle = calculate_angle_3points(r_hp, r_kn, r_ak)

    body_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

    return GeometricFeatures(
        torso_angle_deg=torso_angle_deg,
        aspect_ratio=aspect_ratio,
        relative_hip_height=relative_hip_height,
        left_knee_flexion_deg=l_knee_angle,
        right_knee_flexion_deg=r_knee_angle,
        mean_visibility=mean_vis,
        body_center=body_center,
        bbox=(min_x, min_y, max_x, max_y),
    )
