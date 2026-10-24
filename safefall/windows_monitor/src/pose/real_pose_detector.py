"""
SafeFall Real-Time Optical Pose Estimator with Visual Landmark Overlay.
Utilizes YOLOv8-Pose to detect 17 human anatomical keypoints from optical camera frames,
maps them to normalized biomechanical coordinates, and draws high-contrast visual
skeleton overlays for real-time verification.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from .landmarks import Point3D, GeometricFeatures, extract_geometric_features

# COCO 17 Keypoint Indices
NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

# Skeleton connection pairs for drawing
SKELETON_PAIRS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
]


class RealPoseDetector:
    def __init__(self, model_path: str = "yolov8n-pose.pt", conf_threshold: float = 0.35, imgsz: int = 320):
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._model = None
        self._load_error = None
        self.show_landmarks = True  # Enabled by default per user request

        # Resolve model path across run locations (root vs windows_monitor)
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", model_path)),
                os.path.abspath(os.path.join(os.getcwd(), "safefall", "windows_monitor", model_path)),
                os.path.abspath(os.path.join(os.getcwd(), model_path)),
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_path = c
                    break

        try:
            from ultralytics import YOLO
            self._model = YOLO(resolved_path)
            print(f"[SafeFall Pose] Successfully initialized YOLO pose model from {resolved_path} (imgsz={imgsz})")
        except Exception as e:
            self._load_error = str(e)
            print(f"[SafeFall Pose] Warning: Could not initialize YOLO model: {e}")

    def is_available(self) -> bool:
        return self._model is not None

    def detect_landmarks_raw(
        self, frame_bgr: np.ndarray, imgsz: Optional[int] = None
    ) -> Tuple[Optional[List[Point3D]], Optional[np.ndarray]]:
        """
        Runs fast pose inference without drawing overhead.
        Returns:
            (landmarks_33_list, kpts_array_17x3)
        """
        if self._model is None or frame_bgr is None:
            return None, None

        h, w = frame_bgr.shape[:2]
        sz = imgsz or self.imgsz

        try:
            results = self._model(frame_bgr, verbose=False, conf=self.conf_threshold, imgsz=sz)
            if not results or len(results[0].keypoints) == 0:
                return None, None

            kpts_tensor = results[0].keypoints.data[0]
            kpts = kpts_tensor.cpu().numpy()

            # Map COCO 17 to SafeFall 33 landmark standard
            landmarks_33 = [Point3D(0.5, 0.5, 0.0, 0.0) for _ in range(33)]
            landmarks_33[0] = Point3D(float(kpts[0][0]) / w, float(kpts[0][1]) / h, 0.0, float(kpts[0][2]))
            landmarks_33[11] = Point3D(float(kpts[5][0]) / w, float(kpts[5][1]) / h, 0.0, float(kpts[5][2]))
            landmarks_33[12] = Point3D(float(kpts[6][0]) / w, float(kpts[6][1]) / h, 0.0, float(kpts[6][2]))
            landmarks_33[23] = Point3D(float(kpts[11][0]) / w, float(kpts[11][1]) / h, 0.0, float(kpts[11][2]))
            landmarks_33[24] = Point3D(float(kpts[12][0]) / w, float(kpts[12][1]) / h, 0.0, float(kpts[12][2]))
            landmarks_33[25] = Point3D(float(kpts[13][0]) / w, float(kpts[13][1]) / h, 0.0, float(kpts[13][2]))
            landmarks_33[26] = Point3D(float(kpts[14][0]) / w, float(kpts[14][1]) / h, 0.0, float(kpts[14][2]))
            landmarks_33[27] = Point3D(float(kpts[15][0]) / w, float(kpts[15][1]) / h, 0.0, float(kpts[15][2]))
            landmarks_33[28] = Point3D(float(kpts[16][0]) / w, float(kpts[16][1]) / h, 0.0, float(kpts[16][2]))

            return landmarks_33, kpts
        except Exception as e:
            print(f"[SafeFall Pose Detection Error] {e}")
            return None, None

    def detect_landmarks(
        self, frame_bgr: np.ndarray, draw_overlay: bool = True, imgsz: Optional[int] = None
    ) -> Tuple[Optional[List[Point3D]], np.ndarray]:
        """
        Runs pose detection on the given frame.
        Returns:
            (landmarks_33_list, annotated_frame_bgr)
        """
        if self._model is None or frame_bgr is None:
            return None, frame_bgr

        display_frame = frame_bgr.copy() if draw_overlay else frame_bgr
        landmarks_33, kpts = self.detect_landmarks_raw(frame_bgr, imgsz=imgsz)

        if landmarks_33 is None:
            if draw_overlay and self.show_landmarks:
                cv2.putText(
                    display_frame,
                    "NO PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
            return None, display_frame

        if draw_overlay and self.show_landmarks and kpts is not None:
            h, w = frame_bgr.shape[:2]
            self.draw_skeleton(display_frame, kpts, w, h)

        return landmarks_33, display_frame

    def draw_skeleton(self, frame: np.ndarray, kpts: np.ndarray, w: Optional[int] = None, h: Optional[int] = None):
        """Draws vibrant, high-contrast skeleton bones and joint keypoints."""
        if w is None or h is None:
            h, w = frame.shape[:2]
        self._draw_skeleton(frame, kpts, w, h)

    def _draw_skeleton(self, frame: np.ndarray, kpts: np.ndarray, w: int, h: int):
        """Draws vibrant, high-contrast skeleton bones and joint keypoints."""
        # Draw skeleton connecting lines
        for p1_idx, p2_idx in SKELETON_PAIRS:
            x1, y1, conf1 = kpts[p1_idx]
            x2, y2, conf2 = kpts[p2_idx]

            if conf1 >= self.conf_threshold and conf2 >= self.conf_threshold:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                # Glowing dual-line effect (outer dark border, inner bright neon cyan/yellow)
                cv2.line(frame, pt1, pt2, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw anatomical keypoint circles
        joint_names = {
            5: "L.Sh", 6: "R.Sh",
            11: "L.Hip", 12: "R.Hip",
            13: "L.Knee", 14: "R.Knee",
            15: "L.Ank", 16: "R.Ank"
        }

        for i in range(17):
            x, y, conf = kpts[i]
            if conf >= self.conf_threshold:
                pt = (int(x), int(y))
                # Color code: green for high confidence, orange for moderate
                color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
                cv2.circle(frame, pt, 5, (0, 0, 0), -1, cv2.LINE_AA)  # outer ring
                cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)      # inner filled

                # Label main joints
                if i in joint_names:
                    cv2.putText(
                        frame,
                        f"{joint_names[i]} ({int(conf*100)}%)",
                        (int(x) + 6, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Calculate bounding box
        valid_x = [kpts[i][0] for i in range(17) if kpts[i][2] >= self.conf_threshold]
        valid_y = [kpts[i][1] for i in range(17) if kpts[i][2] >= self.conf_threshold]

        if valid_x and valid_y:
            min_x, max_x = int(min(valid_x)), int(max(valid_x))
            min_y, max_y = int(min(valid_y)), int(max(valid_y))

            # Bounding box
            cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 255, 128), 2)
            cv2.putText(
                frame,
                "TRACKED PERSON [LANDMARKS ON]",
                (min_x - 10, max(20, min_y - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )
