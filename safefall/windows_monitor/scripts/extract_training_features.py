"""
SafeFall LE2I Feature Extractor and Dataset Builder.
Extracts YOLOv8-pose keypoints and 8-dimensional kinematic sliding-window sequences
from the LE2I Fall Detection dataset across Coffee_room_01, Coffee_room_02, Home_01, and Home_02.
Saves the processed sequences with room and sequence metadata for leakage-free model training.
"""

import os
import sys
import glob
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

# Ensure windows_monitor is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.landmarks import extract_geometric_features, Point3D
from src.pose.posture_classifier import PostureClassifier, Posture
from src.detection.temporal_features import KinematicFeatureBuffer
from src.detection.sequence_model import TemporalSequenceEvaluator

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


LABEL_MAP = {
    "NORMAL_MOVEMENT": 0,
    "SITTING_DOWN": 1,
    "GETTING_UP": 2,
    "INTENTIONAL_LYING_DOWN": 3,
    "LYING_OR_SLEEPING": 4,
    "SUSPECTED_FALL": 5,
    "UNCERTAIN": 6,
}


def find_dataset_pairs(training_dir: str) -> List[Dict[str, Any]]:
    """Discovers all paired video and annotation files across dataset rooms."""
    rooms = ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]
    pairs = []

    for room in rooms:
        room_root = os.path.join(training_dir, room, room)
        if not os.path.isdir(room_root):
            room_root = os.path.join(training_dir, room)
            if not os.path.isdir(room_root):
                continue

        ann_folder = os.path.join(room_root, "Annotation_files")
        if not os.path.isdir(ann_folder):
            ann_folder = os.path.join(room_root, "Annotations_files")

        vid_folder = os.path.join(room_root, "Videos")
        if not os.path.isdir(vid_folder):
            vid_folder = os.path.join(room_root, "videos")

        txt_files = sorted(glob.glob(os.path.join(ann_folder, "*.txt")))
        for txt_path in txt_files:
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            vid_path = os.path.join(vid_folder, f"{base_name}.avi")
            if not os.path.exists(vid_path):
                continue

            # Parse annotation file
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                continue

            is_fall = False
            fall_start = 0
            fall_end = 0

            if len(lines) >= 2 and lines[0].isdigit() and lines[1].isdigit():
                s = int(lines[0])
                e = int(lines[1])
                if s > 0 and e > 0:
                    is_fall = True
                    fall_start = s
                    fall_end = e

            pairs.append({
                "room": room,
                "video_path": vid_path,
                "annotation_path": txt_path,
                "video_id": f"{room}_{base_name}",
                "is_fall": is_fall,
                "fall_start": fall_start,
                "fall_end": fall_end,
                "num_lines": len(lines),
            })

    return pairs


def extract_video_features(
    yolo_model: Any,
    evaluator: TemporalSequenceEvaluator,
    posture_classifier: PostureClassifier,
    video_info: Dict[str, Any],
    stride: int = 2,
    batch_size: int = 32,
    imgsz: int = 192,
) -> Tuple[List[List[float]], List[int]]:
    """
    Reads a video, runs batch YOLO pose estimation, computes kinematic derivatives,
    and returns (frame_feature_vectors, frame_labels).
    """
    cap = cv2.VideoCapture(video_info["video_path"])
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames < 10:
        return [], []

    sampled_indices = list(range(0, total_frames, stride))
    sampled_frames = [frames[i] for i in sampled_indices]

    # Run batch pose inference
    results = yolo_model(sampled_frames, imgsz=imgsz, batch=batch_size, verbose=False)

    kinematic_buf = KinematicFeatureBuffer(
        window_size_frames=15,
        sample_rate_hz=25.0 / stride,
    )

    feature_vectors = []
    frame_labels = []
    fall_s = video_info["fall_start"]
    fall_e = video_info["fall_end"]
    is_fall = video_info["is_fall"]

    lying_start_t = None

    for sub_idx, res in enumerate(results):
        orig_idx = sampled_indices[sub_idx]
        t = orig_idx / 25.0
        frame_h, frame_w = sampled_frames[sub_idx].shape[:2]

        if len(res.keypoints) == 0:
            continue

        kpts_tensor = res.keypoints.data[0]
        kpts = kpts_tensor.cpu().numpy()

        # Map COCO 17 to SafeFall 33 landmark standard
        l33 = [Point3D(0.5, 0.5, 0.0, 0.0) for _ in range(33)]
        l33[0] = Point3D(float(kpts[0][0]) / frame_w, float(kpts[0][1]) / frame_h, 0.0, float(kpts[0][2]))
        l33[11] = Point3D(float(kpts[5][0]) / frame_w, float(kpts[5][1]) / frame_h, 0.0, float(kpts[5][2]))
        l33[12] = Point3D(float(kpts[6][0]) / frame_w, float(kpts[6][1]) / frame_h, 0.0, float(kpts[6][2]))
        l33[23] = Point3D(float(kpts[11][0]) / frame_w, float(kpts[11][1]) / frame_h, 0.0, float(kpts[11][2]))
        l33[24] = Point3D(float(kpts[12][0]) / frame_w, float(kpts[12][1]) / frame_h, 0.0, float(kpts[12][2]))
        l33[25] = Point3D(float(kpts[13][0]) / frame_w, float(kpts[13][1]) / frame_h, 0.0, float(kpts[13][2]))
        l33[26] = Point3D(float(kpts[14][0]) / frame_w, float(kpts[14][1]) / frame_h, 0.0, float(kpts[14][2]))
        l33[27] = Point3D(float(kpts[15][0]) / frame_w, float(kpts[15][1]) / frame_h, 0.0, float(kpts[15][2]))
        l33[28] = Point3D(float(kpts[16][0]) / frame_w, float(kpts[16][1]) / frame_h, 0.0, float(kpts[16][2]))

        geom = extract_geometric_features(l33)
        metrics = kinematic_buf.add_frame(t, geom, l33)
        if not metrics or not geom:
            continue

        posture, _ = posture_classifier.classify(geom)

        # 8-dimensional normalized feature vector
        vec = evaluator.encode_feature_vector(
            torso_angle=metrics.current_torso_angle,
            hip_height=metrics.current_hip_height,
            aspect_ratio=metrics.current_aspect_ratio,
            left_knee=geom.left_knee_flexion_deg,
            right_knee=geom.right_knee_flexion_deg,
            downward_vel=metrics.downward_hip_velocity,
            angular_vel=metrics.angular_velocity_deg_s,
            motion_energy=metrics.motion_energy,
        )

        # Determine label
        if is_fall:
            if orig_idx < (fall_s - 8):
                label_name = "SITTING_DOWN" if posture == Posture.SITTING else "NORMAL_MOVEMENT"
            elif (fall_s - 8) <= orig_idx <= (fall_e + 4):
                label_name = "SUSPECTED_FALL"
            else:
                label_name = "LYING_OR_SLEEPING"
        else:
            # ADL video
            if posture == Posture.LYING or metrics.current_torso_angle > 50.0 or metrics.current_aspect_ratio > 1.15:
                if lying_start_t is None:
                    lying_start_t = t
                if (t - lying_start_t) < 1.8:
                    label_name = "INTENTIONAL_LYING_DOWN"
                else:
                    label_name = "LYING_OR_SLEEPING"
            elif posture == Posture.SITTING:
                lying_start_t = None
                label_name = "SITTING_DOWN"
            else:
                lying_start_t = None
                label_name = "NORMAL_MOVEMENT"

        feature_vectors.append(vec)
        frame_labels.append(LABEL_MAP[label_name])

    return feature_vectors, frame_labels


def create_sliding_windows(
    vectors: List[List[float]],
    labels: List[int],
    window_size: int = 16,
    fall_step: int = 2,
    normal_step: int = 4,
) -> Tuple[List[List[List[float]]], List[int]]:
    """Creates fixed-length sliding windows with step sampling."""
    windows = []
    win_labels = []

    if len(vectors) < window_size:
        return [], []

    i = 0
    while i <= len(vectors) - window_size:
        win_vecs = vectors[i : i + window_size]
        win_lbls = labels[i : i + window_size]

        # Prioritize key dynamic events in the window
        fall_code = LABEL_MAP["SUSPECTED_FALL"]
        intentional_lying_code = LABEL_MAP["INTENTIONAL_LYING_DOWN"]
        lying_code = LABEL_MAP["LYING_OR_SLEEPING"]

        if win_lbls.count(fall_code) >= 3:
            assigned_label = fall_code
            step = fall_step
        elif win_lbls.count(intentional_lying_code) >= 3:
            assigned_label = intentional_lying_code
            step = fall_step
        elif win_lbls.count(lying_code) >= (window_size // 2):
            assigned_label = lying_code
            step = normal_step
        else:
            # Majority label
            vals, counts = np.unique(win_lbls, return_counts=True)
            assigned_label = int(vals[np.argmax(counts)])
            step = normal_step

        windows.append(win_vecs)
        win_labels.append(assigned_label)
        i += step

    return windows, win_labels


def main():
    print("=" * 65)
    print("  [SafeFall] LE2I Dataset Feature Extraction Pipeline")
    print("=" * 65)

    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "training"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "le2i_extracted_dataset.npz")

    if not os.path.isdir(training_dir):
        print(f"Error: Training directory not found at {training_dir}")
        return

    pairs = find_dataset_pairs(training_dir)
    print(f"Found {len(pairs)} matched videos across 4 rooms.")
    fall_cnt = sum(1 for p in pairs if p["is_fall"])
    adl_cnt = len(pairs) - fall_cnt
    print(f"  - Falls: {fall_cnt}")
    print(f"  - ADLs : {adl_cnt}")

    yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov8n-pose.pt"))
    if not os.path.exists(yolo_model_path):
        yolo_model_path = "yolov8n-pose.pt"

    print(f"Loading YOLO pose model from {yolo_model_path}...")
    yolo = YOLO(yolo_model_path)
    evaluator = TemporalSequenceEvaluator()
    posture_classifier = PostureClassifier()

    all_windows = []
    all_labels = []
    all_rooms = []
    all_video_ids = []

    t0_all = time.time()
    for idx, p in enumerate(pairs):
        t0 = time.time()
        vecs, lbls = extract_video_features(
            yolo_model=yolo,
            evaluator=evaluator,
            posture_classifier=posture_classifier,
            video_info=p,
            stride=2,
            batch_size=32,
            imgsz=192,
        )

        wins, win_lbls = create_sliding_windows(vecs, lbls, window_size=16)
        all_windows.extend(wins)
        all_labels.extend(win_lbls)
        all_rooms.extend([p["room"]] * len(wins))
        all_video_ids.extend([p["video_id"]] * len(wins))

        elapsed = time.time() - t0
        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            print(f"[{idx+1:03d}/{len(pairs):03d}] {p['video_id']:<30} | {len(wins):3d} windows | {elapsed:.2f}s | Total Windows: {len(all_windows)}")

    total_time = time.time() - t0_all
    print("=" * 65)
    print(f"Extraction complete in {total_time:.1f}s ({total_time/60.0:.2f} min).")
    print(f"Total Sliding Windows: {len(all_windows)}")

    # Class distribution
    lbl_arr = np.array(all_labels)
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    for code in sorted(list(set(all_labels))):
        cnt = np.sum(lbl_arr == code)
        print(f"  Class {code} ({inv_map[code]}): {cnt} ({cnt/len(all_labels)*100:.1f}%)")

    np_windows = np.array(all_windows, dtype=np.float32)
    np_labels = np.array(all_labels, dtype=np.int64)
    np_rooms = np.array(all_rooms)
    np_video_ids = np.array(all_video_ids)

    print(f"Saving extracted dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        sequences=np_windows,
        labels=np_labels,
        rooms=np_rooms,
        video_ids=np_video_ids,
    )
    print(f"Dataset successfully saved ({os.path.getsize(output_path)/(1024*1024):.2f} MB).")


if __name__ == "__main__":
    main()
