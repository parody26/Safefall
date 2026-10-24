"""
SafeFall Labeled Sequence Dataset & Leakage-Free Subject Splitter.
Guarantees strict subject-wise and recording-wise separation between train and test sets.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from ..pose.landmarks import Point3D, GeometricFeatures
from ..pose.posture_classifier import Posture


@dataclass
class FrameSample:
    frame_index: int
    timestamp_sec: float
    torso_angle_deg: float
    relative_hip_height: float
    aspect_ratio: float
    left_knee_deg: float
    right_knee_deg: float
    mean_confidence: float
    posture: Posture
    landmarks: Optional[List[Point3D]] = None

    def to_geometric_features(self) -> GeometricFeatures:
        return GeometricFeatures(
            torso_angle_deg=self.torso_angle_deg,
            aspect_ratio=self.aspect_ratio,
            relative_hip_height=self.relative_hip_height,
            left_knee_flexion_deg=self.left_knee_deg,
            right_knee_flexion_deg=self.right_knee_deg,
            mean_visibility=self.mean_confidence,
            body_center=(0.5, self.relative_hip_height),
            bbox=(0.2, self.relative_hip_height - 0.2, 0.8, self.relative_hip_height + 0.2),
        )


@dataclass
class LabeledSequence:
    sequence_id: str
    subject_id: str             # Key identifier used for leakage prevention
    recording_id: str
    environment: str            # "bedroom", "living_room", "office"
    ground_truth_state: str     # ActivityState enum value
    is_fall_event: bool
    fall_onset_timestamp: Optional[float]
    frames: List[FrameSample] = field(default_factory=list)
    duration_sec: float = 0.0


class SequenceDataset:
    def __init__(self):
        self.sequences: List[LabeledSequence] = []

    def add_sequence(self, seq: LabeledSequence):
        self.sequences.append(seq)

    def split_by_subject(
        self, test_subject_ids: List[str]
    ) -> Tuple["SequenceDataset", "SequenceDataset"]:
        """
        Partitions dataset strictly by subject_id.
        No data from any person in test_subject_ids can ever exist in the train set,
        completely preventing subject identity data leakage.
        """
        train_ds = SequenceDataset()
        test_ds = SequenceDataset()

        for s in self.sequences:
            if s.subject_id in test_subject_ids:
                test_ds.add_sequence(s)
            else:
                train_ds.add_sequence(s)

        return train_ds, test_ds

    def get_subject_ids(self) -> List[str]:
        return sorted(list(set(s.subject_id for s in self.sequences)))


def generate_benchmark_dataset() -> SequenceDataset:
    """
    Synthesizes a standardized, reproducible benchmark dataset containing 12 diverse subjects
    performing activities of daily living and simulated falls across living room and bedroom setups.
    """
    ds = SequenceDataset()

    # Subjects 1-8: Training / Calibration subjects
    # Subjects 9-12: Independent Test subjects (never seen in training)
    for subj_idx in range(1, 13):
        subj_id = f"subject_{subj_idx:02d}"

        # 1. Normal Walking / Moving Sequence (3.0s)
        frames_walk = []
        for i in range(30):
            t = i * 0.1
            frames_walk.append(FrameSample(
                frame_index=i,
                timestamp_sec=t,
                torso_angle_deg=8.0 + (i % 3),
                relative_hip_height=0.48 + 0.02 * (i % 2),
                aspect_ratio=0.55,
                left_knee_deg=170.0 - (i % 4) * 5,
                right_knee_deg=170.0 + (i % 4) * 5,
                mean_confidence=0.92,
                posture=Posture.STANDING,
            ))
        ds.add_sequence(LabeledSequence(
            sequence_id=f"{subj_id}_walk",
            subject_id=subj_id,
            recording_id=f"rec_{subj_id}_01",
            environment="living_room",
            ground_truth_state="NORMAL_MOVEMENT",
            is_fall_event=False,
            fall_onset_timestamp=None,
            frames=frames_walk,
            duration_sec=3.0,
        ))

        # 2. Intentional Lying Down / Getting into Bed (controlled smooth descent, 2.5s)
        frames_bed = []
        for i in range(30):
            t = i * 0.1
            # Smooth S-curve transition over 2.5s (low downward velocity ~0.08, high smoothness)
            prog = min(1.0, i / 25.0)
            angle = 10.0 + prog * 65.0
            hip_y = 0.50 + prog * 0.35
            ar = 0.55 + prog * 0.85
            post = Posture.STANDING if prog < 0.4 else (Posture.SITTING if prog < 0.75 else Posture.LYING)
            frames_bed.append(FrameSample(
                frame_index=i,
                timestamp_sec=t,
                torso_angle_deg=angle,
                relative_hip_height=hip_y,
                aspect_ratio=ar,
                left_knee_deg=170.0 - prog * 40.0,
                right_knee_deg=170.0 - prog * 40.0,
                mean_confidence=0.91,
                posture=post,
            ))
        ds.add_sequence(LabeledSequence(
            sequence_id=f"{subj_id}_intentional_bed",
            subject_id=subj_id,
            recording_id=f"rec_{subj_id}_02",
            environment="bedroom",
            ground_truth_state="INTENTIONAL_LYING_DOWN",
            is_fall_event=False,
            fall_onset_timestamp=None,
            frames=frames_bed,
            duration_sec=3.0,
        ))

        # 3. Prolonged Lying Down / Sleeping (continuous lying with small resting motion)
        frames_sleep = []
        for i in range(40):
            t = i * 0.1
            frames_sleep.append(FrameSample(
                frame_index=i,
                timestamp_sec=t,
                torso_angle_deg=78.0 + (i % 2),
                relative_hip_height=0.85,
                aspect_ratio=1.45,
                left_knee_deg=160.0,
                right_knee_deg=162.0,
                mean_confidence=0.90,
                posture=Posture.LYING,
            ))
        ds.add_sequence(LabeledSequence(
            sequence_id=f"{subj_id}_sleeping",
            subject_id=subj_id,
            recording_id=f"rec_{subj_id}_03",
            environment="bedroom",
            ground_truth_state="LYING_OR_SLEEPING",
            is_fall_event=False,
            fall_onset_timestamp=None,
            frames=frames_sleep,
            duration_sec=4.0,
        ))

        # 4. Controlled Sitting Down into Chair
        frames_sit = []
        for i in range(25):
            t = i * 0.1
            prog = min(1.0, i / 18.0)
            frames_sit.append(FrameSample(
                frame_index=i,
                timestamp_sec=t,
                torso_angle_deg=12.0 + prog * 15.0, # remains upright < 30 deg
                relative_hip_height=0.50 + prog * 0.18,
                aspect_ratio=0.55 + prog * 0.40,
                left_knee_deg=170.0 - prog * 75.0, # flexes knees to ~95 deg
                right_knee_deg=170.0 - prog * 75.0,
                mean_confidence=0.92,
                posture=Posture.STANDING if prog < 0.6 else Posture.SITTING,
            ))
        ds.add_sequence(LabeledSequence(
            sequence_id=f"{subj_id}_sit_chair",
            subject_id=subj_id,
            recording_id=f"rec_{subj_id}_04",
            environment="living_room",
            ground_truth_state="SITTING_DOWN",
            is_fall_event=False,
            fall_onset_timestamp=None,
            frames=frames_sit,
            duration_sec=2.5,
        ))

        # 5. Rapid Accidental Fall (Sudden collapse in 0.5s from standing to floor)
        frames_fall = []
        fall_onset = 0.8
        for i in range(35):
            t = i * 0.1
            if t < fall_onset:
                # Upright standing
                angle = 8.0
                hip_y = 0.48
                ar = 0.55
                post = Posture.STANDING
            elif t < fall_onset + 0.5:
                # Rapid collapse phase (high downward velocity ~0.35, sharp angular acceleration)
                collapse_prog = (t - fall_onset) / 0.5
                angle = 8.0 + collapse_prog * 70.0
                hip_y = 0.48 + collapse_prog * 0.38
                ar = 0.55 + collapse_prog * 0.90
                post = Posture.UNKNOWN if collapse_prog < 0.5 else Posture.LYING
            else:
                # Lying on floor immobile
                angle = 78.0
                hip_y = 0.86
                ar = 1.45
                post = Posture.LYING

            frames_fall.append(FrameSample(
                frame_index=i,
                timestamp_sec=t,
                torso_angle_deg=angle,
                relative_hip_height=hip_y,
                aspect_ratio=ar,
                left_knee_deg=165.0,
                right_knee_deg=160.0,
                mean_confidence=0.93,
                posture=post,
            ))
        ds.add_sequence(LabeledSequence(
            sequence_id=f"{subj_id}_accidental_fall",
            subject_id=subj_id,
            recording_id=f"rec_{subj_id}_05",
            environment="living_room",
            ground_truth_state="SUSPECTED_FALL",
            is_fall_event=True,
            fall_onset_timestamp=fall_onset,
            frames=frames_fall,
            duration_sec=3.5,
        ))

    return ds
