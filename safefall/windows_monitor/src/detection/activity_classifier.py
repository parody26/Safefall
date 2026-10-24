"""
SafeFall Multi-State Activity & Dynamic Fall Classification Engine.
Distinguishes accidental falls from intentional lying down, sleeping, sitting down,
and getting into bed through temporal kinematic trajectory analysis.
"""

from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from ..pose.landmarks import GeometricFeatures
from ..pose.posture_classifier import Posture
from .temporal_features import TemporalKinematicMetrics
from .sequence_model import TemporalSequenceEvaluator


class ActivityState(str, Enum):
    NORMAL_MOVEMENT = "NORMAL_MOVEMENT"
    SITTING_DOWN = "SITTING_DOWN"
    GETTING_UP = "GETTING_UP"
    INTENTIONAL_LYING_DOWN = "INTENTIONAL_LYING_DOWN"
    LYING_OR_SLEEPING = "LYING_OR_SLEEPING"
    SUSPECTED_FALL = "SUSPECTED_FALL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class CameraCalibration:
    camera_placement: str = "table_level"      # "table_level" (0.8-1.2m), "eye_level" (1.5m), "ceiling_mount" (2.2m+)
    fall_velocity_threshold: float = 0.14       # Normalized vertical hip displacement per sec
    angular_velocity_threshold: float = 35.0    # Degrees tilt change per sec
    impact_acceleration_threshold: float = 0.22 # Downward acceleration spike (norm/sec^2)
    intentional_descent_max_vel: float = 0.11   # Below this, descent is controlled
    min_confidence_threshold: float = 0.35      # Below this, state is UNCERTAIN

    @classmethod
    def for_placement(cls, placement: str) -> "CameraCalibration":
        if placement == "ceiling_mount":
            # From ceiling, downward vertical movement appears foreshortened, but bounding box shrinks/flattens
            return cls(
                camera_placement=placement,
                fall_velocity_threshold=0.09,
                angular_velocity_threshold=28.0,
                impact_acceleration_threshold=0.15,
                intentional_descent_max_vel=0.07,
            )
        elif placement == "eye_level":
            return cls(
                camera_placement=placement,
                fall_velocity_threshold=0.16,
                angular_velocity_threshold=38.0,
                impact_acceleration_threshold=0.25,
                intentional_descent_max_vel=0.12,
            )
        else: # table_level default
            return cls(
                camera_placement="table_level",
                fall_velocity_threshold=0.14,
                angular_velocity_threshold=35.0,
                impact_acceleration_threshold=0.22,
                intentional_descent_max_vel=0.11,
            )


@dataclass
class ActivityEvaluationResult:
    activity_state: ActivityState
    confidence: float
    trigger_reason: str
    is_fall_candidate: bool
    kinematics: Optional[TemporalKinematicMetrics] = None


class DynamicActivityClassifier:
    def __init__(
        self,
        calibration: Optional[CameraCalibration] = None,
        sequence_evaluator: Optional[TemporalSequenceEvaluator] = None,
    ):
        self.calib = calibration or CameraCalibration()
        self.sequence_evaluator = sequence_evaluator or TemporalSequenceEvaluator()
        self.sequence_buffer: deque = deque(maxlen=16)

        self.last_state: ActivityState = ActivityState.NORMAL_MOVEMENT
        self.lying_start_timestamp: Optional[float] = None
        self.prior_posture: Posture = Posture.STANDING
        self.prior_posture_timestamp: float = 0.0

    def evaluate(
        self,
        current_posture: Posture,
        posture_conf: float,
        metrics: Optional[TemporalKinematicMetrics],
        timestamp: Optional[float] = None,
        min_lying_confirmation_sec: float = 0.3,
        max_transition_duration_sec: float = 2.2,
        features: Optional[GeometricFeatures] = None,
    ) -> ActivityEvaluationResult:
        """
        Classifies temporal dynamics across consecutive frames into one of the 7 states:
        NORMAL_MOVEMENT, SITTING_DOWN, GETTING_UP, INTENTIONAL_LYING_DOWN,
        LYING_OR_SLEEPING, SUSPECTED_FALL, UNCERTAIN.
        """
        now = metrics.timestamp if metrics else (timestamp or 0.0)

        # Buffer feature vector for sequence GRU inference
        if metrics is not None:
            lk = features.left_knee_flexion_deg if features else 170.0
            rk = features.right_knee_flexion_deg if features else 170.0
            vec = self.sequence_evaluator.encode_feature_vector(
                torso_angle=metrics.current_torso_angle,
                hip_height=metrics.current_hip_height,
                aspect_ratio=metrics.current_aspect_ratio,
                left_knee=lk,
                right_knee=rk,
                downward_vel=metrics.downward_hip_velocity,
                angular_vel=metrics.angular_velocity_deg_s,
                motion_energy=metrics.motion_energy,
            )
            self.sequence_buffer.append(vec)

        # 1. Update prior standing / sitting posture timestamp regardless of metrics availability
        if current_posture in (Posture.STANDING, Posture.SITTING) and posture_conf >= self.calib.min_confidence_threshold:
            self.prior_posture = current_posture
            self.prior_posture_timestamp = now
            self.lying_start_timestamp = None

        # 2. Check for low confidence or occluded tracking
        if (
            posture_conf < self.calib.min_confidence_threshold
            or current_posture == Posture.UNKNOWN
        ):
            self.last_state = ActivityState.UNCERTAIN
            return ActivityEvaluationResult(
                activity_state=ActivityState.UNCERTAIN,
                confidence=posture_conf,
                trigger_reason="Low landmark confidence or body occlusion",
                is_fall_candidate=False,
                kinematics=metrics,
            )

        # If buffer is still priming (less than 2 frames), assume normal movement for standing/sitting
        if metrics is None:
            if current_posture in (Posture.STANDING, Posture.SITTING):
                self.last_state = ActivityState.NORMAL_MOVEMENT
                return ActivityEvaluationResult(
                    activity_state=ActivityState.NORMAL_MOVEMENT,
                    confidence=posture_conf,
                    trigger_reason=f"Calibrating baseline kinematics ({current_posture.value})",
                    is_fall_candidate=False,
                    kinematics=None,
                )
            self.last_state = ActivityState.UNCERTAIN
            return ActivityEvaluationResult(
                activity_state=ActivityState.UNCERTAIN,
                confidence=posture_conf,
                trigger_reason="Accumulating frame buffer",
                is_fall_candidate=False,
                kinematics=None,
            )

        # 3. GETTING UP
        # Negative downward velocity = upward motion
        if metrics.downward_hip_velocity < -0.05 and current_posture in (Posture.STANDING, Posture.SITTING):
            self.last_state = ActivityState.GETTING_UP
            self.lying_start_timestamp = None
            return ActivityEvaluationResult(
                activity_state=ActivityState.GETTING_UP,
                confidence=0.88,
                trigger_reason=f"Upward vertical velocity ({metrics.downward_hip_velocity:.2f} norm/s)",
                is_fall_candidate=False,
                kinematics=metrics,
            )

        # 4. SITTING DOWN
        # Downward movement ending in upright sitting posture
        if current_posture == Posture.SITTING and metrics.current_torso_angle < 50.0:
            self.last_state = ActivityState.SITTING_DOWN
            return ActivityEvaluationResult(
                activity_state=ActivityState.SITTING_DOWN,
                confidence=0.90,
                trigger_reason=f"Controlled descent into chair/sofa (torso angle: {metrics.current_torso_angle:.1f}°)",
                is_fall_candidate=False,
                kinematics=metrics,
            )

        # 5. LYING POSTURE EVALUATION (INTENTIONAL vs SLEEPING vs SUSPECTED FALL)
        if current_posture == Posture.LYING:
            if self.lying_start_timestamp is None:
                self.lying_start_timestamp = now

            lying_duration = max(0.0, now - self.lying_start_timestamp)

            # Wait for minimum lying confirmation duration (e.g. 0.3s) before declaring fall or rest
            if lying_duration < (min_lying_confirmation_sec - 0.05):
                return ActivityEvaluationResult(
                    activity_state=ActivityState.NORMAL_MOVEMENT,
                    confidence=posture_conf,
                    trigger_reason=f"Confirming lying posture ({lying_duration:.2f}s < {min_lying_confirmation_sec:.2f}s)",
                    is_fall_candidate=False,
                    kinematics=metrics,
                )

            # Safeguard: Prolonged resting or sleeping on surface
            # Applies when the subject has been lying down continuously for >= 2.0s
            # with low motion energy, and was not flagged as a suspected fall candidate.
            if (
                lying_duration >= 2.0
                and metrics.motion_energy < 0.04
                and self.last_state != ActivityState.SUSPECTED_FALL
            ):
                self.last_state = ActivityState.LYING_OR_SLEEPING
                return ActivityEvaluationResult(
                    activity_state=ActivityState.LYING_OR_SLEEPING,
                    confidence=round(posture_conf, 2),
                    trigger_reason=f"Resting/sleeping on surface (lying duration: {lying_duration:.1f}s, motion: {metrics.motion_energy:.3f})",
                    is_fall_candidate=False,
                    kinematics=metrics,
                )

            # Check transition duration from prior upright posture
            time_since_prior = now - self.prior_posture_timestamp
            if time_since_prior > max_transition_duration_sec:
                self.last_state = ActivityState.INTENTIONAL_LYING_DOWN
                return ActivityEvaluationResult(
                    activity_state=ActivityState.INTENTIONAL_LYING_DOWN,
                    confidence=0.88,
                    trigger_reason=f"Slow settling into bed/floor ({time_since_prior:.1f}s > {max_transition_duration_sec:.1f}s limit)",
                    is_fall_candidate=False,
                    kinematics=metrics,
                )

            # Evaluate peak kinematics during the transition window
            peak_v = metrics.peak_downward_velocity
            peak_w = metrics.peak_angular_velocity
            peak_a = metrics.peak_downward_acceleration

            # Neural sequence discriminator check (when model weights are loaded)
            neural_disc = None
            if self.sequence_evaluator.is_trained and len(self.sequence_buffer) >= 6:
                neural_disc = self.sequence_evaluator.distinguish_fall_vs_lying(list(self.sequence_buffer))

            # Kinematic fall indicators:
            has_rapid_velocity = peak_v >= self.calib.fall_velocity_threshold       # >= 0.14
            has_high_velocity = peak_v >= 0.38                                     # Rapid vertical drop (free-fall)
            has_abrupt_tilt = peak_w >= self.calib.angular_velocity_threshold       # >= 35 deg/s
            has_high_tilt = peak_w >= 65.0                                         # Rapid tumbling/snap
            is_rough_descent = metrics.trajectory_smoothness <= 0.40
            is_smooth_recline = metrics.trajectory_smoothness >= 0.50

            # If subject was sitting, reclining onto bed/sofa is a safe seated transfer unless high drop velocity exists
            is_seated_recline = (self.prior_posture == Posture.SITTING and peak_v < 0.38)

            # Distinct fall dynamics:
            # Accidental falls exhibit rough/erratic descent (low smoothness) AND rapid downward velocity or abrupt tilt,
            # or extreme free-fall velocity (peak_v >= 0.45)
            is_fall_dynamics = (
                not is_seated_recline
                and (
                    (peak_v >= 0.45 and has_abrupt_tilt)
                    or (has_rapid_velocity and has_abrupt_tilt and is_rough_descent)
                    or (has_high_velocity and is_rough_descent)
                )
            )

            is_neural_fall = (
                not is_seated_recline
                and neural_disc is not None
                and neural_disc.get("is_fall", False)
                and has_rapid_velocity
                and (is_rough_descent or peak_v >= 0.40)
            )

            # A) Suspected fall evaluation:
            # Evaluated first when clear gravitational collapse kinematics or neural fall signature exists
            if is_fall_dynamics or is_neural_fall:
                self.last_state = ActivityState.SUSPECTED_FALL
                overall_conf = round(
                    min(0.98, max(0.70, (peak_v / self.calib.fall_velocity_threshold) * 0.5 + (peak_w / self.calib.angular_velocity_threshold) * 0.5)),
                    2,
                )
                if is_neural_fall and neural_disc:
                    overall_conf = max(overall_conf, neural_disc["confidence"])

                trigger = (
                    f"AI Sequence Model + kinetic collapse: peak vel={peak_v:.2f}, peak ang={peak_w:.1f}°, fall_prob={neural_disc['fall_probability']:.2f}"
                    if is_neural_fall and neural_disc
                    else f"Abrupt kinetic collapse: peak vel={peak_v:.2f} norm/s, peak ang={peak_w:.1f}°/s, impact accel={peak_a:.2f}"
                )
                return ActivityEvaluationResult(
                    activity_state=ActivityState.SUSPECTED_FALL,
                    confidence=overall_conf,
                    trigger_reason=trigger,
                    is_fall_candidate=True,
                    kinematics=metrics,
                )

            # B) Intentional lying down evaluation:
            # Controlled, gentle recline without gravitational collapse kinematics
            # 1. Neural sequence model verified intentional lying (with safe kinematics)
            if (
                neural_disc is not None
                and neural_disc.get("is_lying", False)
                and not is_fall_dynamics
                and peak_v < 0.28
                and peak_a < (self.calib.impact_acceleration_threshold * 1.2)
            ):
                self.last_state = ActivityState.INTENTIONAL_LYING_DOWN
                trigger = (
                    f"AI Sequence Model verified intentional lying "
                    f"(p_lying={neural_disc['lying_probability']:.2f}, p_fall={neural_disc['fall_probability']:.2f})"
                )
                return ActivityEvaluationResult(
                    activity_state=ActivityState.INTENTIONAL_LYING_DOWN,
                    confidence=neural_disc["confidence"],
                    trigger_reason=trigger,
                    is_fall_candidate=False,
                    kinematics=metrics,
                )

            # 2. Rule-based controlled recline / descent checks
            has_no_impact_shock = peak_a < self.calib.impact_acceleration_threshold
            is_smooth_recline = (
                metrics.trajectory_smoothness >= 0.42
                and has_no_impact_shock
                and peak_v < 0.24
                and peak_w < 55.0
            )
            is_slow_descent = peak_v <= self.calib.intentional_descent_max_vel and has_no_impact_shock
            is_seated_transfer = self.prior_posture == Posture.SITTING and peak_v < 0.20 and has_no_impact_shock

            if is_smooth_recline or is_slow_descent or is_seated_transfer:
                self.last_state = ActivityState.INTENTIONAL_LYING_DOWN
                trigger = f"Controlled intentional descent (peak vel: {peak_v:.2f}, smooth: {metrics.trajectory_smoothness:.2f}, impact accel: {peak_a:.2f})"
                return ActivityEvaluationResult(
                    activity_state=ActivityState.INTENTIONAL_LYING_DOWN,
                    confidence=0.92,
                    trigger_reason=trigger,
                    is_fall_candidate=False,
                    kinematics=metrics,
                )

            # Default for lying without fall kinematics
            self.last_state = ActivityState.INTENTIONAL_LYING_DOWN
            return ActivityEvaluationResult(
                activity_state=ActivityState.INTENTIONAL_LYING_DOWN,
                confidence=0.85,
                trigger_reason="Lying posture reached without kinetic collapse dynamics",
                is_fall_candidate=False,
                kinematics=metrics,
            )

        # 6. NORMAL MOVEMENT
        self.last_state = ActivityState.NORMAL_MOVEMENT
        return ActivityEvaluationResult(
            activity_state=ActivityState.NORMAL_MOVEMENT,
            confidence=round(posture_conf, 2),
            trigger_reason=f"Routine activity ({current_posture.value}, torso: {metrics.current_torso_angle:.1f}°, motion: {metrics.motion_energy:.3f})",
            is_fall_candidate=False,
            kinematics=metrics,
        )

    def reset(self):
        self.last_state = ActivityState.NORMAL_MOVEMENT
        self.lying_start_timestamp = None
        self.prior_posture = Posture.STANDING
        self.prior_posture_timestamp = 0.0
        self.sequence_buffer.clear()

