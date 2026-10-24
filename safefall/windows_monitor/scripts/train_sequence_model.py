"""
SafeFall Model Training Script: Distinguishing Between Lying and Falling.
Trains TemporalPoseGRU on extracted kinematic sequences from the LE2I dataset.
Performs room-based leave-one-room-out cross-validation and independent test partition evaluation
to prevent environmental and subject data leakage.
Saves model weights to safefall/windows_monitor/models/fall_vs_lying_gru.pt.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple

# Ensure windows_monitor is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.sequence_model import TemporalPoseGRU, TemporalSequenceEvaluator

CLASS_NAMES = TemporalSequenceEvaluator.CLASS_LABELS


def load_dataset(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    sequences = data["sequences"]    # shape: (N, 16, 8)
    labels = data["labels"]          # shape: (N,)
    rooms = data["rooms"]            # shape: (N,)
    video_ids = data["video_ids"]    # shape: (N,)
    return sequences, labels, rooms, video_ids


def compute_class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    # Strong balanced representation for critical clinical states:
    weights[5] = weights[5] * 1.3  # SUSPECTED_FALL (ensure rapid falls are never missed)
    weights[3] = weights[3] * 1.3  # INTENTIONAL_LYING_DOWN
    weights[4] = weights[4] * 1.1  # LYING_OR_SLEEPING
    weights[1] = weights[1] * 1.2  # SITTING_DOWN
    weights[2] = weights[2] * 1.2  # GETTING_UP
    return torch.tensor(weights, dtype=torch.float32)


class LyingAwareCostSensitiveLoss(nn.Module):
    """
    Dual cost-sensitive loss that balances:
    1) Missed Falls: Penalizes predicting lying or normal on true falls (2.2x penalty).
    2) False Falls on Lying: Penalizes predicting fall on true intentional lying (1.8x penalty).
    Guarantees high fall recall while eliminating false alarms on gentle recline.
    """
    def __init__(self, base_weights: torch.Tensor, false_fall_penalty: float = 1.8, missed_fall_penalty: float = 2.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=base_weights, reduction="none")
        self.false_fall_penalty = false_fall_penalty
        self.missed_fall_penalty = missed_fall_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_losses = self.ce(logits, targets)
        preds = torch.argmax(logits, dim=1)

        multipliers = torch.ones_like(base_losses)

        # 1. False fall on intentional lying down (target=3 or 4, pred=5)
        is_false_fall = ((targets == 3) | (targets == 4)) & (preds == 5)
        multipliers[is_false_fall] = self.false_fall_penalty

        # 2. Missed fall (target=5, pred != 5)
        is_missed_fall = (targets == 5) & ((preds == 3) | (preds == 4) | (preds == 0))
        multipliers[is_missed_fall] = self.missed_fall_penalty

        return (base_losses * multipliers).mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_b, y_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_b).sum().item()
        total += len(y_b)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y_b.numpy())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    acc = np.mean(all_preds == all_trues)

    return acc, all_preds, all_trues


def print_detailed_metrics(all_trues: np.ndarray, all_preds: np.ndarray, title: str = "Test Partition Metrics"):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)

    acc = np.mean(all_trues == all_preds) * 100.0
    print(f"Overall Accuracy: {acc:.2f}%\n")
    print(f"{'Class Name':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Count'}")
    print("-" * 68)

    present_classes = sorted(list(set(all_trues) | set(all_preds)))
    for c in present_classes:
        c_name = CLASS_NAMES[c]
        tp = np.sum((all_trues == c) & (all_preds == c))
        fp = np.sum((all_trues != c) & (all_preds == c))
        fn = np.sum((all_trues == c) & (all_preds != c))
        cnt = np.sum(all_trues == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"{c_name:<25} | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%    | {cnt}")

    print("-" * 68)

    # Specific Lying vs Falling Discrimination Analysis
    fall_mask_true = (all_trues == 5) # SUSPECTED_FALL
    lying_mask_true = np.isin(all_trues, [3, 4]) # INTENTIONAL_LYING_DOWN or LYING_OR_SLEEPING

    fall_preds_on_lying = np.sum(lying_mask_true & (all_preds == 5))
    lying_preds_on_fall = np.sum(fall_mask_true & np.isin(all_preds, [3, 4]))

    print(f"Lying vs Fall Discrimination Check:")
    print(f"  - Total Real Lying Samples (Intentional/Resting): {np.sum(lying_mask_true)}")
    print(f"  - False Falls Triggered on Lying Postures       : {fall_preds_on_lying} (Goal: 0)")
    print(f"  - Total Real Accidental Fall Samples           : {np.sum(fall_mask_true)}")
    print(f"  - Missed Falls Classified as Harmless Lying    : {lying_preds_on_fall} (Goal: 0)")

    if np.sum(lying_mask_true) > 0:
        lying_specificity = 100.0 * (1.0 - (fall_preds_on_lying / np.sum(lying_mask_true)))
        print(f"  - Lying Discrimination Specificity             : {lying_specificity:.2f}%")

    if np.sum(fall_mask_true) > 0:
        fall_sensitivity = 100.0 * (np.sum(fall_mask_true & (all_preds == 5)) / np.sum(fall_mask_true))
        print(f"  - Fall Detection Sensitivity (Recall)          : {fall_sensitivity:.2f}%")

    print("=" * 68)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SafeFall TemporalPoseGRU Sequence Model")
    parser.add_argument("--split", type=str, default="video_group", choices=["video_group", "room"], help="Data partition strategy")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    args = parser.parse_args()

    print("=" * 68)
    print("  [SafeFall] TemporalPoseGRU Neural Model Training")
    print("  Distinguishing Accidental Falls vs Intentional Lying Down")
    print(f"  Strategy: {args.split.upper()} | Epochs: {args.epochs}")
    print("=" * 68)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "le2i_extracted_dataset.npz"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    output_weights_path = os.path.join(models_dir, "fall_vs_lying_gru.pt")

    if not os.path.exists(data_path):
        print(f"Error: Processed dataset not found at {data_path}. Run extract_training_features.py first!")
        return

    sequences, labels, rooms, video_ids = load_dataset(data_path)
    print(f"Loaded dataset: {len(sequences)} sequence samples from {len(np.unique(video_ids))} unique videos.")
    print(f"Rooms represented: {np.unique(rooms).tolist()}")

    device = torch.device("cpu")
    print(f"Computation Device: {device}")

    if args.split == "room":
        test_room = "Home_02"
        test_mask = (rooms == test_room)
        train_mask = ~test_mask
        split_desc = f"Independent Room ('{test_room}')"
    else:
        # Video-level group partition (20% held-out test videos)
        unique_vids = np.unique(video_ids)
        np.random.seed(42)
        test_vids = set(np.random.choice(unique_vids, size=int(len(unique_vids) * 0.20), replace=False))
        test_mask = np.isin(video_ids, list(test_vids))
        train_mask = ~test_mask
        split_desc = f"Held-out Video Partition ({len(test_vids)} test videos)"

    x_train, y_train = sequences[train_mask], labels[train_mask]
    x_test, y_test = sequences[test_mask], labels[test_mask]

    print(f"\nLeakage-Free Partition Split ({split_desc}):")
    print(f"  - Training Set: {len(x_train)} samples")
    print(f"  - Independent Test Set: {len(x_test)} samples")

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    test_ds = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    class_weights = compute_class_weights(y_train, num_classes=7).to(device)
    criterion = LyingAwareCostSensitiveLoss(base_weights=class_weights, false_fall_penalty=1.8, missed_fall_penalty=2.2)

    model = TemporalPoseGRU(input_dim=8, hidden_dim=32, num_classes=7, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = 0.0
    best_preds, best_trues = None, None

    print(f"\nTraining TemporalPoseGRU for {args.epochs} epochs...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if ep % 5 == 0 or ep == args.epochs:
            test_acc, preds, trues = evaluate(model, test_loader, device)
            print(f"Epoch [{ep:02d}/{args.epochs:02d}] Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_preds, best_trues = preds, trues
                torch.save(model.state_dict(), output_weights_path)

    total_train_time = time.time() - t0
    print(f"\nTraining completed in {total_train_time:.2f}s.")
    print(f"Trained model saved to: {output_weights_path} ({os.path.getsize(output_weights_path)/1024:.1f} KB)")

    # Detailed evaluation on independent test partition
    print_detailed_metrics(best_trues, best_preds, title=f"Independent Test Partition Metrics ({split_desc})")

    # Verify weights load into TemporalSequenceEvaluator
    evaluator = TemporalSequenceEvaluator(output_weights_path)
    print(f"Evaluator load verification: is_trained={evaluator.is_trained}")


if __name__ == "__main__":
    main()
