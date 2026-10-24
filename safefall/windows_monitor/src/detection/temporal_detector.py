"""
SafeFall Enhanced Temporal Fall Detector.
Combines sliding-window kinematic derivations (downward velocity, angular rate, impact acceleration,
trajectory smoothness) and explicit dynamic activity classification to discriminate accidental falls
from intentional bed entry, resting/sleeping, bending, and controlled sitting.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..pose.landmarks import GeometricFeatures, Point3D
from ..pose.posture_classifier import Posture
from .temporal_features import KinematicFeatureBuffer, TemporalKinematicMetrics
from .activity_classifier import (
    ActivityState,
    CameraCalibration,
    DynamicActivityClassifier,
    ActivityEvaluationResult,
)


@dataclass
class Observation:
    timestamp: float               # monotonic seconds
    posture: Posture
    confidence: float
    features: Optional[GeometricFeatures]


@dataclass
class CandidateFallEvent:
    detected_at: float
    transition_type: str          # "STANDING_TO_LYING" | "SITTING_TO_LYING" | "ACCIDENTAL_COLLAPSE"
    duration_sec: float
    velocity_hip: float           # vertical displacement per second
    angular_velocity: float       # degrees per second
    confidence: float
    trigger_reason: str = ""
    activity_state: str = "SUSPECTED_FALL"
    kinematics: Optional[TemporalKinematicMetrics] = None


class TemporalFallDetector:
    def __init__(
        self,
        history_window_sec: float = 4.0,
        max_transition_duration_sec: float = 2.2,
        min_lying_confirmation_sec: float = 0.3,
        cooldown_sec: float = 12.0,
        calibration: Optional[CameraCalibration] = None,
    ):
        self.history_window_sec = history_window_sec
        self.max_transition_duration_sec = max_transition_duration_sec
        self.min_lying_confirmation_sec = min_lying_confirmation_sec
        self.cooldown_sec = cooldown_sec

        self.calib = calibration or CameraCalibration()
        self.kinematic_buffer = KinematicFeatureBuffer(window_size_frames=30)
        self.activity_classifier = DynamicActivityClassifier(calibration=self.calib)

        self.history: deque[Observation] = deque()
        self.last_detection_timestamp: float = -9999.0
        self.last_activity_result: Optional[ActivityEvaluationResult] = None
        self._is_post_fall_active: bool = False
        self._upright_stable_start: Optional[float] = None
        self._min_upright_recovery_sec: float = 6.0

    def add_observation(
        self,
        posture: Posture,
        confidence: float,
        features: Optional[GeometricFeatures],
        timestamp: Optional[float] = None,
        landmarks: Optional[List[Point3D]] = None,
    ) -> Optional[CandidateFallEvent]:
        """
        Ingests a timestamped posture observation.
        Extracts temporal kinematics, evaluates activity state, and returns a CandidateFallEvent
        ONLY when an accidental fall signature is verified.
        Strictly enforces single-incident per fall until full upright recovery.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        obs = Observation(timestamp=now, posture=posture, confidence=confidence, features=features)
        self.history.append(obs)

        # Prune old observations outside history window
        while self.history and (now - self.history[0].timestamp) > self.history_window_sec:
            self.history.popleft()

        # Update kinematic derivatives
        metrics = self.kinematic_buffer.add_frame(now, features, landmarks)

        # Evaluate through dynamic activity state classifier
        eval_result = self.activity_classifier.evaluate(
            current_posture=posture,
            posture_conf=confidence,
            metrics=metrics,
            timestamp=now,
            min_lying_confirmation_sec=self.min_lying_confirmation_sec,
            max_transition_duration_sec=self.max_transition_duration_sec,
            features=features,
        )
        self.last_activity_result = eval_result

        # Post-Fall Lock: While subject remains on ground or shifting, suppress duplicate falls
        if self._is_post_fall_active:
            if posture in (Posture.STANDING, Posture.SITTING) and confidence >= self.calib.min_confidence_threshold:
                if self._upright_stable_start is None:
                    self._upright_stable_start = now
                elif (now - self._upright_stable_start) >= self._min_upright_recovery_sec:
                    # Verified continuous upright recovery! Re-arm fall detection
                    self._is_post_fall_active = False
                    self._upright_stable_start = None
            else:
                self._upright_stable_start = None
            return None

        # Check cooldown to prevent duplicate alerts for the same continuous event
        if (now - self.last_detection_timestamp) < self.cooldown_sec:
            return None

        # Explicit safeguard (Item 5): Avoid triggering a fall simply because a person remains
        # lying down or motionless. A prolonged lying posture must NOT independently count as a fall.
        if eval_result.activity_state == ActivityState.LYING_OR_SLEEPING:
            return None

        # Explicit safeguard: Intentional lying down, sitting down, and normal movements are rejected
        if eval_result.activity_state in (
            ActivityState.INTENTIONAL_LYING_DOWN,
            ActivityState.SITTING_DOWN,
            ActivityState.GETTING_UP,
            ActivityState.NORMAL_MOVEMENT,
            ActivityState.UNCERTAIN,
        ):
            return None

        # Fall candidate detected by activity state machine
        if eval_result.is_fall_candidate:
            self.last_detection_timestamp = now
            self._is_post_fall_active = True
            self._upright_stable_start = None
            transition_type = f"{self.activity_classifier.prior_posture.value}_TO_LYING"
            duration = max(0.2, now - self.activity_classifier.prior_posture_timestamp)

            vel = metrics.peak_downward_velocity if metrics else 0.15
            ang_vel = metrics.peak_angular_velocity if metrics else 45.0

            return CandidateFallEvent(
                detected_at=now,
                transition_type=transition_type,
                duration_sec=round(duration, 2),
                velocity_hip=round(vel, 3),
                angular_velocity=round(ang_vel, 1),
                confidence=eval_result.confidence,
                trigger_reason=eval_result.trigger_reason,
                activity_state=eval_result.activity_state.value,
                kinematics=metrics,
            )

        return None

    def reset(self):
        """Clears observation history, kinematic buffers, and resets cooldown."""
        self.history.clear()
        self.kinematic_buffer.clear()
        self.activity_classifier.reset()
        self.last_detection_timestamp = -9999.0
        self.last_activity_result = None
        self._is_post_fall_active = False
        self._upright_stable_start = None
