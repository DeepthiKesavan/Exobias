from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_pos", "recall_pos", "f1_pos", "log_loss", "brier", "ece", "roc_auc", "pr_auc",
]


def _clip_normalise(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-7, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Top-label ECE with equal-width bins. Returns (ece, per-bin table)."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    rows, ece, n = [], 0.0, len(y_true)
    for b in range(n_bins):
        mask = bin_id == b
        count = int(mask.sum())
        if count:
            acc, conf = correct[mask].mean(), confidence[mask].mean()
            ece += count / n * abs(acc - conf)
        else:
            acc = conf = np.nan
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": count,
                     "mean_confidence": conf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


def compute_metrics(y_true, proba, n_classes: int, positive_index: Optional[int] = None) -> dict:
    """All scalar metrics for one set of predictions. Never raises on degenerate input."""
    y_true = np.asarray(y_true, dtype=int)
    proba = _clip_normalise(proba)
    y_pred = proba.argmax(axis=1)
    binary = n_classes == 2
    positive_index = 1 if positive_index is None else positive_index
    out = {"n_samples": int(len(y_true))}
    if len(y_true) == 0:
        return {**out, **{k: np.nan for k in METRIC_COLUMNS}}

    labels = list(range(n_classes))
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["precision_macro"] = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        out["recall_macro"] = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        out["f1_macro"] = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        if binary:
            out["precision_pos"] = precision_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
            out["recall_pos"] = recall_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
            out["f1_pos"] = f1_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
        else:
            out["precision_pos"] = out["recall_pos"] = out["f1_pos"] = np.nan

    out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    if binary:
        out["brier"] = float(brier_score_loss((y_true == positive_index).astype(int), proba[:, positive_index]))
    else:
        out["brier"] = multiclass_brier(y_true, proba)
    out["ece"] = expected_calibration_error(y_true, proba)[0]

    out["roc_auc"] = out["pr_auc"] = np.nan
    if len(np.unique(y_true)) == n_classes:
        try:
            if binary:
                pos = proba[:, positive_index]
                out["roc_auc"] = float(roc_auc_score(y_true == positive_index, pos))
                out["pr_auc"] = float(average_precision_score(y_true == positive_index, pos))
            else:
                out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            pass
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def classification_report_df(y_true, proba, class_names: list[str]) -> pd.DataFrame:
    y_pred = np.asarray(proba).argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
            output_dict=True, zero_division=0,
        )
    return pd.DataFrame(report).T


def confusion_matrix_df(y_true, proba, class_names: list[str]) -> pd.DataFrame:
    y_pred = np.asarray(proba).argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=[f"True {c}" for c in class_names],
                        columns=[f"Pred {c}" for c in class_names])


def roc_curve_data(y_true, proba, positive_index: int = 1):
    y_bin = (np.asarray(y_true) == positive_index).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    fpr, tpr, _ = roc_curve(y_bin, np.asarray(proba)[:, positive_index])
    return fpr, tpr


def pr_curve_data(y_true, proba, positive_index: int = 1):
    y_bin = (np.asarray(y_true) == positive_index).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    precision, recall, _ = precision_recall_curve(y_bin, np.asarray(proba)[:, positive_index])
    return recall, precision, float(y_bin.mean())


def reliability_curve_data(y_true, proba, positive_index: int = 1, n_bins: int = 10):
    """Positive-class reliability diagram data (binary tasks): (mean predicted, observed fraction)."""
    y_bin = (np.asarray(y_true) == positive_index).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    frac, mean_pred = calibration_curve(y_bin, np.asarray(proba)[:, positive_index],
                                        n_bins=n_bins, strategy="uniform")
    return mean_pred, frac


def overfitting_observations(history: dict, best_epoch: int, patience: int) -> list[str]:
    """Plain-language observations from an ANN learning curve (pure Python, no TF needed)."""
    notes: list[str] = []
    loss, val_loss = history.get("loss", []), history.get("val_loss", [])
    if not loss or not val_loss:
        return ["No learning-curve data available."]
    epochs_run = len(loss)
    b = max(0, min(best_epoch - 1, epochs_run - 1))
    gap = val_loss[b] - loss[b]
    notes.append(
        f"Best epoch was {best_epoch} of {epochs_run} run; validation loss {val_loss[b]:.4f} "
        f"vs training loss {loss[b]:.4f} at that epoch (gap {gap:+.4f})."
    )
    if epochs_run - best_epoch >= patience:
        notes.append(f"Training stopped early: validation loss did not improve for {patience} epochs.")
    elif best_epoch == epochs_run:
        notes.append("Validation loss was still improving at the final epoch; more epochs might help.")
    if gap > 0.05 * max(val_loss[b], 1e-9) and gap > 0.02:
        notes.append("Validation loss is noticeably above training loss at the best epoch: mild overfitting signal.")
    elif gap < -0.02:
        notes.append("Validation loss is below training loss (dropout is active during training loss, "
                     "or the validation set is easier); this is not itself a problem.")
    else:
        notes.append("Training and validation losses are close at the best epoch: no strong overfitting signal.")
    if epochs_run > best_epoch and val_loss[-1] > val_loss[b] * 1.05:
        notes.append("Validation loss rose after the best epoch, so early stopping / best-weight restoration mattered.")
    return notes
