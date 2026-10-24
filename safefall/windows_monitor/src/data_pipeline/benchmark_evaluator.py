"""
SafeFall Benchmark Evaluator & Metrics Reporter.
Computes Precision, Recall, F1, False Alarms per Hour (FAR/hr), Missed Falls,
and Detection Latency across leakage-free test partitions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .sequence_dataset import SequenceDataset, LabeledSequence
from ..detection.temporal_detector import TemporalFallDetector


@dataclass
class EvaluationMetrics:
    total_sequences: int
    fall_sequences_count: int
    adl_sequences_count: int
    total_adl_monitoring_hours: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int        # Missed falls

    precision: float
    recall: float               # Sensitivity
    specificity: float
    f1_score: float
    false_alarms_per_hour: float
    mean_detection_latency_ms: float
    max_detection_latency_ms: float


class BenchmarkEvaluator:
    def __init__(self, detector: Optional[TemporalFallDetector] = None):
        self.detector = detector or TemporalFallDetector()

    def evaluate_dataset(self, dataset: SequenceDataset) -> EvaluationMetrics:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        latencies_ms: List[float] = []
        total_adl_sec = 0.0

        for seq in dataset.sequences:
            self.detector.reset()
            detected_fall = None

            for frame in seq.frames:
                geom = frame.to_geometric_features()
                ev = self.detector.add_observation(
                    posture=frame.posture,
                    confidence=frame.mean_confidence,
                    features=geom,
                    timestamp=frame.timestamp_sec,
                    landmarks=frame.landmarks,
                )
                if ev is not None and detected_fall is None:
                    detected_fall = ev

            if seq.is_fall_event:
                if detected_fall is not None:
                    tp += 1
                    onset = seq.fall_onset_timestamp or 0.0
                    lat_ms = max(0.0, (detected_fall.detected_at - onset) * 1000.0)
                    latencies_ms.append(lat_ms)
                else:
                    fn += 1
            else:
                total_adl_sec += seq.duration_sec
                if detected_fall is not None:
                    fp += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        adl_hours = total_adl_sec / 3600.0
        far_per_hr = (fp / adl_hours) if adl_hours > 0 else 0.0

        mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        max_lat = max(latencies_ms) if latencies_ms else 0.0

        return EvaluationMetrics(
            total_sequences=len(dataset.sequences),
            fall_sequences_count=tp + fn,
            adl_sequences_count=tn + fp,
            total_adl_monitoring_hours=round(adl_hours, 4),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            f1_score=round(f1, 4),
            false_alarms_per_hour=round(far_per_hr, 2),
            mean_detection_latency_ms=round(mean_lat, 1),
            max_detection_latency_ms=round(max_lat, 1),
        )

    def print_report(self, metrics: EvaluationMetrics, title: str = "Benchmark Evaluation Report"):
        print("=" * 68)
        print(f"  [SafeFall] {title}")
        print("=" * 68)
        print(f"Total Evaluated Sequences   : {metrics.total_sequences}")
        print(f"  - Ground Truth Falls      : {metrics.fall_sequences_count}")
        print(f"  - Activities of Daily Life: {metrics.adl_sequences_count}")
        print(f"  - ADL Monitoring Duration : {metrics.total_adl_monitoring_hours * 60:.1f} minutes")
        print("-" * 68)
        print(f"True Positives (TP)         : {metrics.true_positives}")
        print(f"False Positives (FP)        : {metrics.false_positives}")
        print(f"True Negatives (TN)         : {metrics.true_negatives}")
        print(f"Missed Falls (FN)           : {metrics.false_negatives}")
        print("-" * 68)
        print(f"Precision                   : {metrics.precision * 100:.2f}%")
        print(f"Recall (Sensitivity)        : {metrics.recall * 100:.2f}%")
        print(f"Specificity                 : {metrics.specificity * 100:.2f}%")
        print(f"F1-Score                    : {metrics.f1_score * 100:.2f}%")
        print(f"False Alarms per Hour (FAR) : {metrics.false_alarms_per_hour} / hr")
        print(f"Mean Detection Latency      : {metrics.mean_detection_latency_ms} ms")
        print(f"Max Detection Latency       : {metrics.max_detection_latency_ms} ms")
        print("=" * 68)
