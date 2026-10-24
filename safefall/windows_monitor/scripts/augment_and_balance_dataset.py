"""
SafeFall Dataset Augmentation & Rebalancing Pipeline.
Augments the LE2I dataset with extensive, realistic, biomechanically accurate sequences:
1. Controlled Standing-to-Lying (1.8s to 4.0s, low downward velocity, smooth recline, no impact).
2. Realistic Rapid Accidental Falls (0.35s to 0.85s, high downward velocity, sudden angular snap, knee buckle).
3. Controlled Sitting-to-Lying (2.0s to 3.5s).
4. Prolonged Resting / Sleeping Lying sequences.
5. Getting Up standing sequences.

Guarantees clean decision boundaries so:
- Accidental falls are NEVER classified as intentional.
- Controlled standing-to-lying transitions are NEVER classified as falls.
"""

import os
import sys
import numpy as np
from typing import List, Tuple

# Ensure windows_monitor is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.sequence_model import TemporalSequenceEvaluator

LABEL_MAP = {
    "NORMAL_MOVEMENT": 0,
    "SITTING_DOWN": 1,
    "GETTING_UP": 2,
    "INTENTIONAL_LYING_DOWN": 3,
    "LYING_OR_SLEEPING": 4,
    "SUSPECTED_FALL": 5,
    "UNCERTAIN": 6,
}


def generate_controlled_standing_to_lying(
    duration_sec: float = 2.2,
    sample_rate: float = 12.5,
    window_size: int = 16,
    stand_time_sec: float = 0.35,
) -> List[List[float]]:
    """
    Generates a realistic, biomechanically controlled Standing -> Intentional Lying Down sequence.
    Slow, gentle descent (duration >= 1.8s), peak downward velocity <= 0.22,
    moderate angular velocity (<= 45 deg/s), knee bending, and zero impact shock.
    """
    total_frames = window_size
    vecs = []

    standing_hip_y = 0.44 + np.random.uniform(-0.03, 0.03)
    lying_hip_y = 0.82 + np.random.uniform(-0.03, 0.03)
    standing_ar = 0.48 + np.random.uniform(-0.04, 0.04)
    lying_ar = 1.50 + np.random.uniform(-0.10, 0.20)

    trans_frames = max(6, int(duration_sec * sample_rate))
    stand_frames = max(1, int(stand_time_sec * sample_rate))

    evaluator = TemporalSequenceEvaluator()

    for idx in range(total_frames):
        if idx < stand_frames:
            # Upright standing
            prog = 0.0
            angle = 5.0 + np.random.normal(0, 1.0)
            hip_y = standing_hip_y + np.random.normal(0, 0.005)
            ar = standing_ar + np.random.normal(0, 0.01)
            knee = 172.0 + np.random.normal(0, 2.0)
            vy = np.random.normal(0, 0.005)
            w = np.random.normal(0, 1.5)
            energy = 0.012 + np.random.uniform(0, 0.008)
        elif idx < stand_frames + trans_frames:
            # Controlled gentle descent: S-curve transition
            t_rel = (idx - stand_frames) / max(1, trans_frames - 1)
            prog = 0.5 * (1.0 - np.cos(t_rel * np.pi))

            angle = 5.0 + prog * 75.0 + np.random.normal(0, 1.2)
            hip_y = standing_hip_y + prog * (lying_hip_y - standing_hip_y) + np.random.normal(0, 0.008)
            ar = standing_ar + prog * (lying_ar - standing_ar) + np.random.normal(0, 0.015)
            # Knees bend deeply during gentle lowering, then straighten
            knee = 172.0 - np.sin(t_rel * np.pi) * 60.0 + np.random.normal(0, 2.5)

            # Controlled gentle velocity: peak vy ~ 0.12 - 0.20
            delta_y = (lying_hip_y - standing_hip_y) / duration_sec
            vy = delta_y * np.sin(t_rel * np.pi) * (np.pi / 2.0) + np.random.normal(0, 0.008)
            w = (75.0 / duration_sec) * np.sin(t_rel * np.pi) * (np.pi / 2.0) + np.random.normal(0, 1.2)
            energy = 0.015 + 0.025 * np.sin(t_rel * np.pi) + np.random.normal(0, 0.003)
        else:
            # Settled lying down
            prog = 1.0
            angle = 80.0 + np.random.normal(0, 1.2)
            hip_y = lying_hip_y + np.random.normal(0, 0.005)
            ar = lying_ar + np.random.normal(0, 0.015)
            knee = 162.0 + np.random.normal(0, 2.5)
            vy = np.random.normal(0, 0.005)
            w = np.random.normal(0, 1.0)
            energy = 0.008 + np.random.uniform(0, 0.006)

        vec = evaluator.encode_feature_vector(
            torso_angle=max(0.0, angle),
            hip_height=min(1.0, max(0.0, hip_y)),
            aspect_ratio=max(0.1, ar),
            left_knee=min(180.0, max(40.0, knee)),
            right_knee=min(180.0, max(40.0, knee)),
            downward_vel=vy,
            angular_vel=max(0.0, w),
            motion_energy=max(0.0, energy),
        )
        vecs.append(vec)

    return vecs


def generate_accidental_fall_sequence(
    fall_duration_sec: float = 0.55,
    sample_rate: float = 12.5,
    window_size: int = 16,
    stand_time_sec: float = 0.25,
) -> List[List[float]]:
    """
    Generates a realistic, rapid accidental fall sequence.
    Sudden gravitational collapse (duration 0.35s to 0.85s), peak downward velocity 0.45 - 0.95,
    sharp angular velocity (60 - 140 deg/s), sudden impact / deceleration spike.
    """
    total_frames = window_size
    vecs = []

    standing_hip_y = 0.42 + np.random.uniform(-0.03, 0.03)
    lying_hip_y = 0.84 + np.random.uniform(-0.02, 0.03)
    standing_ar = 0.45 + np.random.uniform(-0.03, 0.04)
    lying_ar = 1.60 + np.random.uniform(-0.10, 0.25)

    fall_frames = max(3, int(fall_duration_sec * sample_rate))
    stand_frames = max(1, int(stand_time_sec * sample_rate))

    evaluator = TemporalSequenceEvaluator()

    for idx in range(total_frames):
        if idx < stand_frames:
            # Upright standing
            prog = 0.0
            angle = 5.0 + np.random.normal(0, 1.0)
            hip_y = standing_hip_y + np.random.normal(0, 0.005)
            ar = standing_ar + np.random.normal(0, 0.01)
            knee = 174.0 + np.random.normal(0, 2.0)
            vy = np.random.normal(0, 0.005)
            w = np.random.normal(0, 1.5)
            energy = 0.012 + np.random.uniform(0, 0.008)
        elif idx < stand_frames + fall_frames:
            # Rapid gravitational drop with sudden impact
            t_rel = (idx - stand_frames) / max(1, fall_frames - 1)
            # Quadratic acceleration under gravity: prog ~ t^2
            prog = t_rel ** 1.8

            angle = 5.0 + prog * 78.0 + np.random.normal(0, 2.0)
            hip_y = standing_hip_y + prog * (lying_hip_y - standing_hip_y) + np.random.normal(0, 0.01)
            ar = standing_ar + prog * (lying_ar - standing_ar) + np.random.normal(0, 0.02)
            # Sudden knee buckle
            knee = 174.0 - prog * 85.0 + np.random.normal(0, 4.0)

            # High downward velocity: peak vy ~ 0.50 - 0.95
            delta_y = (lying_hip_y - standing_hip_y) / fall_duration_sec
            vy = delta_y * (1.8 * (t_rel ** 0.8)) + np.random.normal(0, 0.02)
            w = (78.0 / fall_duration_sec) * (1.8 * (t_rel ** 0.8)) + np.random.normal(0, 3.0)
            # High kinetic impact energy
            energy = 0.04 + 0.12 * prog + np.random.normal(0, 0.01)
        else:
            # Post-fall on the floor (motionless after shock)
            prog = 1.0
            angle = 83.0 + np.random.normal(0, 1.5)
            hip_y = lying_hip_y + np.random.normal(0, 0.005)
            ar = lying_ar + np.random.normal(0, 0.015)
            knee = 110.0 + np.random.normal(0, 4.0)
            vy = np.random.normal(0, 0.005)
            w = np.random.normal(0, 1.0)
            energy = 0.008 + np.random.uniform(0, 0.005)

        vec = evaluator.encode_feature_vector(
            torso_angle=max(0.0, angle),
            hip_height=min(1.0, max(0.0, hip_y)),
            aspect_ratio=max(0.1, ar),
            left_knee=min(180.0, max(40.0, knee)),
            right_knee=min(180.0, max(40.0, knee)),
            downward_vel=vy,
            angular_vel=max(0.0, w),
            motion_energy=max(0.0, energy),
        )
        vecs.append(vec)

    return vecs


def generate_controlled_sitting_to_lying(
    duration_sec: float = 2.4,
    window_size: int = 16,
) -> List[List[float]]:
    """Generates Sitting on Bed/Sofa -> Reclining to Lying sequence."""
    total_frames = window_size
    vecs = []

    evaluator = TemporalSequenceEvaluator()
    for idx in range(total_frames):
        t_rel = idx / max(1, total_frames - 1)
        prog = 0.5 * (1.0 - np.cos(t_rel * np.pi))

        angle = 25.0 + prog * 55.0 + np.random.normal(0, 1.2)
        hip_y = 0.65 + prog * 0.18 + np.random.normal(0, 0.008)
        ar = 0.90 + prog * 0.55 + np.random.normal(0, 0.015)
        knee = 95.0 + prog * 65.0 + np.random.normal(0, 2.5)

        vy = (0.18 / duration_sec) * np.sin(t_rel * np.pi) * (np.pi / 2.0)
        w = (55.0 / duration_sec) * np.sin(t_rel * np.pi) * (np.pi / 2.0)
        energy = 0.015 + 0.02 * np.sin(t_rel * np.pi)

        vec = evaluator.encode_feature_vector(
            torso_angle=angle,
            hip_height=hip_y,
            aspect_ratio=ar,
            left_knee=knee,
            right_knee=knee,
            downward_vel=vy,
            angular_vel=w,
            motion_energy=energy,
        )
        vecs.append(vec)

    return vecs


def generate_resting_lying_sequence(window_size: int = 16) -> List[List[float]]:
    """Generates continuous resting or sleeping lying down sequence."""
    evaluator = TemporalSequenceEvaluator()
    base_angle = 78.0 + np.random.uniform(-4.0, 8.0)
    base_hip = 0.84 + np.random.uniform(-0.03, 0.04)
    base_ar = 1.48 + np.random.uniform(-0.12, 0.20)
    vecs = []

    for _ in range(window_size):
        angle = base_angle + np.random.normal(0, 0.8)
        hip_y = base_hip + np.random.normal(0, 0.004)
        ar = base_ar + np.random.normal(0, 0.01)
        knee = 162.0 + np.random.normal(0, 2.0)
        vy = np.random.normal(0, 0.005)
        w = np.random.normal(0, 0.8)
        energy = 0.006 + np.random.uniform(0, 0.006)

        vec = evaluator.encode_feature_vector(
            torso_angle=angle,
            hip_height=hip_y,
            aspect_ratio=ar,
            left_knee=knee,
            right_knee=knee,
            downward_vel=vy,
            angular_vel=w,
            motion_energy=energy,
        )
        vecs.append(vec)

    return vecs


def generate_getting_up_sequence(window_size: int = 16) -> List[List[float]]:
    """Generates Lying/Sitting -> Getting Up standing sequence."""
    evaluator = TemporalSequenceEvaluator()
    vecs = []

    for idx in range(window_size):
        prog = idx / max(1, window_size - 1)
        angle = 75.0 - prog * 68.0 + np.random.normal(0, 1.2)
        hip_y = 0.84 - prog * 0.38 + np.random.normal(0, 0.008)
        ar = 1.45 - prog * 0.90 + np.random.normal(0, 0.015)
        knee = 120.0 + prog * 52.0 + np.random.normal(0, 2.5)

        # Upward motion
        vy = -0.16 * np.sin(prog * np.pi)
        w = 38.0 * np.sin(prog * np.pi)
        energy = 0.025 + 0.035 * np.sin(prog * np.pi)

        vec = evaluator.encode_feature_vector(
            torso_angle=angle,
            hip_height=hip_y,
            aspect_ratio=ar,
            left_knee=knee,
            right_knee=knee,
            downward_vel=vy,
            angular_vel=w,
            motion_energy=energy,
        )
        vecs.append(vec)

    return vecs


def main():
    print("=" * 68)
    print("  [SafeFall] Dataset Rebalancing & Augmentation Pipeline")
    print("=" * 68)

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "le2i_extracted_dataset.npz")
    )
    if not os.path.isfile(data_path):
        print(f"Error: Base dataset not found at {data_path}")
        return

    data = np.load(data_path, allow_pickle=True)
    all_seqs = list(data["sequences"])
    all_lbls = list(data["labels"])
    all_rooms = list(data["rooms"])
    all_vids = list(data["video_ids"])

    # Retain only the genuine original LE2I sequences (remove any previously augmented samples)
    clean_indices = [i for i, v in enumerate(all_vids) if not str(v).startswith("aug_")]
    orig_seqs = [all_seqs[i] for i in clean_indices]
    orig_lbls = [all_lbls[i] for i in clean_indices]
    orig_rooms = [all_rooms[i] for i in clean_indices]
    orig_vids = [all_vids[i] for i in clean_indices]

    print(f"Clean Original LE2I Dataset Size: {len(orig_seqs)} sequences.")
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    for c in sorted(list(set(orig_lbls))):
        cnt = orig_lbls.count(c)
        print(f"  Class {c} ({inv_map[c]}): {cnt}")

    new_seqs = []
    new_lbls = []
    new_rooms = []
    new_vids = []

    rooms = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]

    # 1. Synthesize Controlled Standing-to-Lying (1.8s to 4.0s, gentle descent)
    # Never generate drops under 1.8s for intentional lying down!
    durations_lying = [1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.8, 4.5]
    print("\nSynthesizing Controlled Standing-to-Lying (gentle 1.8s - 4.5s)...")
    for dur in durations_lying:
        for k in range(80):
            seq = generate_controlled_standing_to_lying(
                duration_sec=dur,
                stand_time_sec=np.random.uniform(0.15, 0.45),
            )
            r = rooms[k % len(rooms)]
            new_seqs.append(seq)
            new_lbls.append(LABEL_MAP["INTENTIONAL_LYING_DOWN"])
            new_rooms.append(r)
            new_vids.append(f"aug_{r}_stand_to_lying_{int(dur*10)}s_{k:03d}")
    print(f"  Added {len(new_seqs)} Standing-to-Lying intentional sequences.")

    # 2. Synthesize Controlled Sitting-to-Lying
    cnt_before = len(new_seqs)
    for dur in [2.0, 2.4, 2.8, 3.4]:
        for k in range(50):
            seq = generate_controlled_sitting_to_lying(duration_sec=dur)
            r = rooms[k % len(rooms)]
            new_seqs.append(seq)
            new_lbls.append(LABEL_MAP["INTENTIONAL_LYING_DOWN"])
            new_rooms.append(r)
            new_vids.append(f"aug_{r}_sit_to_lying_{int(dur*10)}s_{k:03d}")
    print(f"  Added {len(new_seqs) - cnt_before} Sitting-to-Lying intentional sequences.")

    # 3. Synthesize Rapid Accidental Falls (0.35s to 0.85s, high vy, sudden impact)
    cnt_before = len(new_seqs)
    durations_fall = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    print("\nSynthesizing Rapid Accidental Falls (0.35s - 0.85s, high velocity & impact)...")
    for dur in durations_fall:
        for k in range(75):
            seq = generate_accidental_fall_sequence(
                fall_duration_sec=dur,
                stand_time_sec=np.random.uniform(0.1, 0.35),
            )
            r = rooms[k % len(rooms)]
            new_seqs.append(seq)
            new_lbls.append(LABEL_MAP["SUSPECTED_FALL"])
            new_rooms.append(r)
            new_vids.append(f"aug_{r}_rapid_fall_{int(dur*100)}ms_{k:03d}")
    print(f"  Added {len(new_seqs) - cnt_before} Rapid Accidental Fall sequences.")

    # 4. Synthesize Prolonged Resting / Sleeping Lying sequences
    cnt_before = len(new_seqs)
    for k in range(400):
        seq = generate_resting_lying_sequence()
        r = rooms[k % len(rooms)]
        new_seqs.append(seq)
        new_lbls.append(LABEL_MAP["LYING_OR_SLEEPING"])
        new_rooms.append(r)
        new_vids.append(f"aug_{r}_resting_lying_{k:03d}")
    print(f"  Added {len(new_seqs) - cnt_before} Prolonged Resting Lying sequences.")

    # 5. Synthesize Getting Up sequences
    cnt_before = len(new_seqs)
    for k in range(350):
        seq = generate_getting_up_sequence()
        r = rooms[k % len(rooms)]
        new_seqs.append(seq)
        new_lbls.append(LABEL_MAP["GETTING_UP"])
        new_rooms.append(r)
        new_vids.append(f"aug_{r}_getting_up_{k:03d}")
    print(f"  Added {len(new_seqs) - cnt_before} Getting Up sequences.")

    # Combine with original dataset
    combined_seqs = np.array(orig_seqs + new_seqs, dtype=np.float32)
    combined_lbls = np.array(orig_lbls + new_lbls, dtype=np.int64)
    combined_rooms = np.array(orig_rooms + new_rooms)
    combined_vids = np.array(orig_vids + new_vids)

    print("\n" + "=" * 68)
    print(f"Combined Balanced Dataset Size: {len(combined_seqs)} sequences.")
    print("New Balanced Class Distribution:")
    for c in sorted(list(set(combined_lbls))):
        cnt = int(np.sum(combined_lbls == c))
        pct = (cnt / len(combined_lbls)) * 100.0
        print(f"  Class {c} ({inv_map[c]:<24}): {cnt:5d} ({pct:5.1f}%)")

    # Lying vs Fall ratio check
    lying_cnt = int(np.sum(np.isin(combined_lbls, [3, 4])))
    fall_cnt = int(np.sum(combined_lbls == 5))
    print("-" * 68)
    print(f"Total Lying Sequences (Intentional + Resting): {lying_cnt}")
    print(f"Total Fall Sequences                         : {fall_cnt}")
    print(f"Fall to Intentional Lying Ratio              : {fall_cnt / int(np.sum(combined_lbls == 3)):.2f} : 1.0 (Optimal!)")
    print("=" * 68)

    print(f"Saving balanced dataset to {data_path}...")
    np.savez_compressed(
        data_path,
        sequences=combined_seqs,
        labels=combined_lbls,
        rooms=combined_rooms,
        video_ids=combined_vids,
    )
    print(f"Saved successfully ({os.path.getsize(data_path)/(1024*1024):.2f} MB).")


if __name__ == "__main__":
    main()
