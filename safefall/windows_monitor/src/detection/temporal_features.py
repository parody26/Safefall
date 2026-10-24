"""
SafeFall Temporal Biomechanical Feature Extraction.
Computes frame-to-frame kinematics, velocities, accelerations, angular rates,
and motion energy across a rolling temporal observation window.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..pose.landmarks import Point3D, GeometricFeatures


@dataclass
class KinematicFrame:
    timestamp: float                    # Monotonic seconds
    torso_angle_deg: float              # 0 = vertical, 90 = horizontal
    relative_hip_height: float          # 0.0 = top of frame, 1.0 = bottom
    aspect_ratio: float                 # Width / Height of bounding envelope
    left_knee_flexion_deg: float
    right_knee_flexion_deg: float
    mean_visibility: float
    body_center: Tuple[float, float]    # (cx, cy)
    landmarks: Optional[List[Point3D]] = None


@dataclass
class TemporalKinematicMetrics:
    timestamp: float
    current_torso_angle: float
    current_hip_height: float
    current_aspect_ratio: float
    
    # First derivatives (velocities)
    downward_hip_velocity: float        # Positive = moving downwards (norm units / sec)
    angular_velocity_deg_s: float       # Change in torso tilt (deg / sec)
    horizontal_velocity: float          # Lateral movement (norm units / sec)
    
    # Second derivatives (accelerations)
    downward_hip_acceleration: float    # Acceleration spike (norm units / sec^2)
    angular_acceleration: float         # Angular acceleration (deg / sec^2)
    
    # Peak windowed metrics (critical for detecting impact collapse during lying confirmation)
    peak_downward_velocity: float
    peak_angular_velocity: float
    peak_downward_acceleration: float

    # Energy and dispersion
    motion_energy: float                # Mean landmark displacement between frames
    trajectory_smoothness: float        # Consistency of downward motion (low variance = controlled)
    time_window_duration: float
    mean_confidence: float


class KinematicFeatureBuffer:
    """
    Maintains a high-resolution sliding window of kinematic observations
    and computes smoothed derivative metrics.
    """
    def __init__(self, window_size_frames: int = 25, sample_rate_hz: float = 12.0):
        self.window_size_frames = window_size_frames
        self.sample_rate_hz = sample_rate_hz
        self.buffer: deque[KinematicFrame] = deque(maxlen=window_size_frames)

    def add_frame(
        self,
        timestamp: float,
        features: Optional[GeometricFeatures],
        landmarks: Optional[List[Point3D]] = None,
    ) -> Optional[TemporalKinematicMetrics]:
        if features is None:
            return None

        frame = KinematicFrame(
            timestamp=timestamp,
            torso_angle_deg=features.torso_angle_deg,
            relative_hip_height=features.relative_hip_height,
            aspect_ratio=features.aspect_ratio,
            left_knee_flexion_deg=features.left_knee_flexion_deg,
            right_knee_flexion_deg=features.right_knee_flexion_deg,
            mean_visibility=features.mean_visibility,
            body_center=features.body_center,
            landmarks=landmarks,
        )
        self.buffer.append(frame)

        if len(self.buffer) < 2:
            return None

        return self.compute_metrics()

    def compute_metrics(self) -> TemporalKinematicMetrics:
        frames = list(self.buffer)
        curr = frames[-1]
        prev = frames[-2]

        dt_instant = max(1e-4, curr.timestamp - prev.timestamp)

        # 1. Instantaneous velocities
        downward_hip_vel_inst = (curr.relative_hip_height - prev.relative_hip_height) / dt_instant
        angular_vel_inst = abs(curr.torso_angle_deg - prev.torso_angle_deg) / dt_instant
        dx = curr.body_center[0] - prev.body_center[0]
        horiz_vel_inst = abs(dx) / dt_instant

        # 2. Windowed velocities (more robust across 5-8 frames, ~0.4 - 0.6 sec)
        lookback_idx = max(0, len(frames) - 6)
        base_frame = frames[lookback_idx]
        dt_window = max(1e-4, curr.timestamp - base_frame.timestamp)
        window_downward_vel = (curr.relative_hip_height - base_frame.relative_hip_height) / dt_window
        window_angular_vel = abs(curr.torso_angle_deg - base_frame.torso_angle_deg) / dt_window

        # 3. Accelerations (second derivatives)
        if len(frames) >= 3:
            prev2 = frames[-3]
            dt_prev = max(1e-4, prev.timestamp - prev2.timestamp)
            prev_downward_vel = (prev.relative_hip_height - prev2.relative_hip_height) / dt_prev
            downward_accel = (downward_hip_vel_inst - prev_downward_vel) / dt_instant

            prev_angular_vel = abs(prev.torso_angle_deg - prev2.torso_angle_deg) / dt_prev
            angular_accel = (angular_vel_inst - prev_angular_vel) / dt_instant
        else:
            downward_accel = 0.0
            angular_accel = 0.0

        # 4. Peak metrics across sliding buffer (preserves descent spike even after landing)
        peak_v = max(0.0, downward_hip_vel_inst, window_downward_vel)
        peak_w = max(angular_vel_inst, window_angular_vel)
        peak_a = max(0.0, downward_accel)

        for i in range(1, len(frames)):
            fa, fb = frames[i - 1], frames[i]
            dt_step = max(1e-4, fb.timestamp - fa.timestamp)
            step_v = (fb.relative_hip_height - fa.relative_hip_height) / dt_step
            step_w = abs(fb.torso_angle_deg - fa.torso_angle_deg) / dt_step
            if step_v > peak_v:
                peak_v = step_v
            if step_w > peak_w:
                peak_w = step_w

        if len(frames) >= 4:
            for i in range(2, len(frames)):
                f0, f1, f2 = frames[i - 2], frames[i - 1], frames[i]
                dt1 = max(1e-4, f1.timestamp - f0.timestamp)
                dt2 = max(1e-4, f2.timestamp - f1.timestamp)
                v1 = (f1.relative_hip_height - f0.relative_hip_height) / dt1
                v2 = (f2.relative_hip_height - f1.relative_hip_height) / dt2
                step_a = (v2 - v1) / dt2
                if step_a > peak_a:
                    peak_a = step_a

        # 5. Motion energy across landmarks
        motion_energy = 0.0
        if curr.landmarks and prev.landmarks and len(curr.landmarks) == len(prev.landmarks):
            displacements = [
                math.hypot(c.x - p.x, c.y - p.y)
                for c, p in zip(curr.landmarks, prev.landmarks)
                if c.visibility > 0.3 and p.visibility > 0.3
            ]
            if displacements:
                motion_energy = sum(displacements) / len(displacements)
        else:
            # Fallback to center and aspect ratio displacement
            motion_energy = math.hypot(
                curr.body_center[0] - prev.body_center[0],
                curr.body_center[1] - prev.body_center[1]
            )

        # 6. Trajectory smoothness / variance across window
        vel_samples = []
        for i in range(1, len(frames)):
            f_a, f_b = frames[i - 1], frames[i]
            dt = max(1e-4, f_b.timestamp - f_a.timestamp)
            vel_samples.append((f_b.relative_hip_height - f_a.relative_hip_height) / dt)

        mean_vel = sum(vel_samples) / len(vel_samples)
        variance = sum((v - mean_vel) ** 2 for v in vel_samples) / len(vel_samples)
        smoothness = 1.0 / (1.0 + variance * 100.0)

        mean_conf = sum(f.mean_visibility for f in frames) / len(frames)
        total_duration = curr.timestamp - frames[0].timestamp

        return TemporalKinematicMetrics(
            timestamp=curr.timestamp,
            current_torso_angle=round(curr.torso_angle_deg, 2),
            current_hip_height=round(curr.relative_hip_height, 3),
            current_aspect_ratio=round(curr.aspect_ratio, 2),
            downward_hip_velocity=round(window_downward_vel, 3),
            angular_velocity_deg_s=round(window_angular_vel, 1),
            horizontal_velocity=round(horiz_vel_inst, 3),
            downward_hip_acceleration=round(downward_accel, 3),
            angular_acceleration=round(angular_accel, 2),
            peak_downward_velocity=round(peak_v, 3),
            peak_angular_velocity=round(peak_w, 1),
            peak_downward_acceleration=round(peak_a, 3),
            motion_energy=round(motion_energy, 4),
            trajectory_smoothness=round(smoothness, 3),
            time_window_duration=round(total_duration, 2),
            mean_confidence=round(mean_conf, 3),
        )

    def clear(self):
        self.buffer.clear()
