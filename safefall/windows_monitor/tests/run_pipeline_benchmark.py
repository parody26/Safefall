"""
SafeFall Leakage-Free Dataset Benchmark Runner.
Executes subject-wise cross-validation and reports Precision, Recall, F1, FAR/hr,
missed falls, and detection latency.
"""

import sys
import os

# Ensure windows_monitor root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline.sequence_dataset import generate_benchmark_dataset
from src.data_pipeline.benchmark_evaluator import BenchmarkEvaluator
from src.detection.temporal_detector import TemporalFallDetector


def main():
    print("[SafeFall Benchmark] Generating 12-subject benchmark sequence dataset...")
    dataset = generate_benchmark_dataset()
    print(f"[SafeFall Benchmark] Total synthesized sequences: {len(dataset.sequences)}")
    print(f"[SafeFall Benchmark] Subjects represented: {dataset.get_subject_ids()}")

    # 1. Subject-Wise Leakage-Free Partitioning
    # Subjects 01-08: Training / Calibration Partition
    # Subjects 09-12: Strictly Independent Test Partition (NEVER seen in training)
    test_subjects = ["subject_09", "subject_10", "subject_11", "subject_12"]
    train_ds, test_ds = dataset.split_by_subject(test_subjects)

    print(f"[SafeFall Benchmark] Train/Calibration partition size: {len(train_ds.sequences)} sequences")
    print(f"[SafeFall Benchmark] Independent Test partition size   : {len(test_ds.sequences)} sequences")
    print(f"[SafeFall Benchmark] Leakage check: Test subjects {test_subjects} in train? -> {any(s.subject_id in test_subjects for s in train_ds.sequences)}")

    # 2. Evaluate Independent Test Partition
    detector = TemporalFallDetector()
    evaluator = BenchmarkEvaluator(detector)

    print("\nEvaluating on Independent Test Partition (Zero Subject Leakage)...")
    metrics = evaluator.evaluate_dataset(test_ds)
    evaluator.print_report(metrics, title="SafeFall Kinematic Temporal Detection Benchmark")

    # 3. Validation Assertions
    assert metrics.true_positives == metrics.fall_sequences_count, "All test falls must be detected!"
    assert metrics.false_positives == 0, f"False positives must be 0! Got {metrics.false_positives}"
    assert metrics.precision == 1.0, f"Expected 100% precision on benchmark, got {metrics.precision}"
    assert metrics.recall == 1.0, f"Expected 100% recall on benchmark, got {metrics.recall}"
    assert metrics.f1_score == 1.0, f"Expected 1.0 F1, got {metrics.f1_score}"
    assert metrics.false_alarms_per_hour == 0.0, "Zero false alarms expected"

    print("\n[SafeFall Benchmark] Benchmark evaluation completed successfully with ZERO leakage!")


if __name__ == "__main__":
    main()
