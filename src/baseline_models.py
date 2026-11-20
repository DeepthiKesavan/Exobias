from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASELINE_NAMES = ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost"]
SCALED_MODELS = {"Logistic Regression"}


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def available_baselines() -> list[str]:
    return [m for m in BASELINE_NAMES if m != "XGBoost" or xgboost_available()]


def uses_scaled_features(name: str) -> bool:
    return name in SCALED_MODELS


def make_model(name: str, seed: int = 42, n_classes: int = 2):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=seed)
    if name == "XGBoost":
        if not xgboost_available():
            raise ImportError("xgboost is not installed (pip install xgboost).")
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        )
    raise ValueError(f"Unknown baseline model '{name}'.")


def train_baseline(name: str, X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                   sample_weight: Optional[np.ndarray] = None, seed: int = 42):
    model = make_model(name, seed, n_classes)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba_aligned(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba with columns guaranteed to be 0..n_classes-1 (missing classes -> 0)."""
    raw = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1]))).astype(int)
    if raw.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return raw
    full = np.zeros((raw.shape[0], n_classes))
    full[:, classes] = raw
    return full
