"""
SafeFall Temporal Sequence Model (GRU / 1D-CNN Architecture).
Processes sliding window sequences of pose kinematic features to predict dynamic activity states.
NOTE: Transparently flags whether model weights are initialized or trained.
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TemporalPoseGRU(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 7, num_layers: int = 2):
        if not HAS_TORCH:
            return
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1D Temporal Convolution feature encoder
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=16,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Permute for Conv1d: (batch_size, input_dim, seq_len)
        x_conv = x.permute(0, 2, 1)
        feat = self.conv(x_conv)
        feat = feat.permute(0, 2, 1) # back to (batch_size, seq_len, 16)

        gru_out, _ = self.gru(feat)
        # Take last time step
        last_step = gru_out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class TemporalSequenceEvaluator:
    """
    Manages temporal sequence inference, feature encoding, and weight checkpointing.
    Transparently reports whether pretrained weights exist or if rule-based fallback is active.
    Provides specialized discrimination between intentional lying and accidental falling.
    """
    CLASS_LABELS = [
        "NORMAL_MOVEMENT",
        "SITTING_DOWN",
        "GETTING_UP",
        "INTENTIONAL_LYING_DOWN",
        "LYING_OR_SLEEPING",
        "SUSPECTED_FALL",
        "UNCERTAIN",
    ]

    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is None:
            default_weights = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "fall_vs_lying_gru.pt")
            )
            if os.path.isfile(default_weights):
                weights_path = default_weights

        self.weights_path = weights_path
        self.is_trained = False
        self.model = None

        if HAS_TORCH:
            self.model = TemporalPoseGRU()
            if weights_path and os.path.isfile(weights_path):
                self.load_weights(weights_path)

    def encode_feature_vector(
        self,
        torso_angle: float,
        hip_height: float,
        aspect_ratio: float,
        left_knee: float,
        right_knee: float,
        downward_vel: float,
        angular_vel: float,
        motion_energy: float,
    ) -> List[float]:
        """Encodes an 8-dimensional normalized frame feature vector."""
        return [
            min(1.0, max(0.0, torso_angle / 90.0)),
            min(1.0, max(0.0, hip_height)),
            min(2.5, max(0.0, aspect_ratio)) / 2.5,
            min(1.0, max(0.0, left_knee / 180.0)),
            min(1.0, max(0.0, right_knee / 180.0)),
            min(1.0, max(-1.0, downward_vel)),
            min(1.0, max(0.0, angular_vel / 100.0)),
            min(1.0, max(0.0, motion_energy * 5.0)),
        ]

    def predict_sequence(self, sequence_vectors: List[List[float]]) -> Tuple[str, float, bool]:
        """
        Runs model inference if trained weights are loaded.
        Returns (predicted_class_name, confidence, is_model_trained).
        """
        if not HAS_TORCH or self.model is None or not self.is_trained:
            return "UNCERTAIN", 0.0, False

        try:
            self.model.eval()
            with torch.no_grad():
                tensor_in = torch.tensor([sequence_vectors], dtype=torch.float32)
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs))
                conf = float(probs[pred_idx])
                return self.CLASS_LABELS[pred_idx], round(conf, 2), True
        except Exception:
            return "UNCERTAIN", 0.0, False

    def predict_probabilities(self, sequence_vectors: List[List[float]]) -> Tuple[Dict[str, float], bool]:
        """
        Returns full class probability distribution across all 7 states.
        """
        if not HAS_TORCH or self.model is None or not self.is_trained:
            return {label: 0.0 for label in self.CLASS_LABELS}, False

        try:
            self.model.eval()
            with torch.no_grad():
                tensor_in = torch.tensor([sequence_vectors], dtype=torch.float32)
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                return {
                    label: float(round(float(probs[idx]), 3))
                    for idx, label in enumerate(self.CLASS_LABELS)
                }, True
        except Exception:
            return {label: 0.0 for label in self.CLASS_LABELS}, False

    def distinguish_fall_vs_lying(self, sequence_vectors: List[List[float]]) -> Dict[str, Any]:
        """
        Specialized discriminator between Accidental Fall and Intentional Lying Down / Resting.
        Combines sequence dynamics to compute clear decision boundaries.
        """
        probs_dict, is_trained = self.predict_probabilities(sequence_vectors)
        if not is_trained:
            return {
                "decision": "UNCERTAIN",
                "is_fall": False,
                "is_lying": False,
                "confidence": 0.0,
                "fall_probability": 0.0,
                "lying_probability": 0.0,
                "is_trained": False,
            }

        fall_prob = probs_dict.get("SUSPECTED_FALL", 0.0)
        intentional_lying_prob = probs_dict.get("INTENTIONAL_LYING_DOWN", 0.0)
        resting_lying_prob = probs_dict.get("LYING_OR_SLEEPING", 0.0)
        normal_prob = probs_dict.get("NORMAL_MOVEMENT", 0.0)
        sitting_prob = probs_dict.get("SITTING_DOWN", 0.0)

        # Distinguish between dynamic transition (FALL vs INTENTIONAL_LYING)
        # 1. Accidental Fall:
        # Fall probability is significant (>= 0.35) and exceeds or matches intentional transition probability
        if fall_prob >= 0.35 and fall_prob >= intentional_lying_prob:
            decision = "FALL"
            is_fall = True
            is_lying = False
            conf = fall_prob
        # 2. Intentional Lying Down:
        # Intentional transition probability is significant (>= 0.30) and exceeds fall probability
        elif intentional_lying_prob >= 0.30 and intentional_lying_prob > fall_prob:
            decision = "LYING"
            is_fall = False
            is_lying = True
            conf = intentional_lying_prob
        # 3. Resting / Sleeping Lying Down (static horizontal posture without fall kinematics)
        elif resting_lying_prob >= 0.40 and fall_prob < 0.25:
            decision = "LYING"
            is_fall = False
            is_lying = True
            conf = resting_lying_prob
        # 4. Sitting Down
        elif sitting_prob > normal_prob and sitting_prob > fall_prob:
            decision = "SITTING"
            is_fall = False
            is_lying = False
            conf = sitting_prob
        else:
            decision = "NORMAL"
            is_fall = False
            is_lying = False
            conf = normal_prob

        return {
            "decision": decision,
            "is_fall": is_fall,
            "is_lying": is_lying,
            "confidence": round(conf, 2),
            "fall_probability": round(fall_prob, 3),
            "lying_probability": round(intentional_lying_prob + resting_lying_prob, 3),
            "is_trained": True,
            "probabilities": probs_dict,
        }

    def load_weights(self, path: str) -> bool:
        if not HAS_TORCH or self.model is None:
            return False
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.is_trained = True
            print(f"[SafeFall GRU] Loaded trained weights from {path}")
            return True
        except Exception as e:
            self.is_trained = False
            print(f"[SafeFall GRU] Notice: Weights file {path} not loaded ({e}). Model is currently untrained.")
            return False

    def save_weights(self, path: str):
        if HAS_TORCH and self.model is not None:
            torch.save(self.model.state_dict(), path)
            self.is_trained = True

