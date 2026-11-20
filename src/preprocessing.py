from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .data_validation import (
    CLASS_ORDER, DEFAULT_LOG_FEATURES, GROUP_COLUMN, NAME_COLUMN, SCORE_COLUMN,
    TARGET_COLUMN, assert_features_allowed, normalise_labels,
)

# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


@dataclass
class CleaningReport:
    removal_log: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    missing_before: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    inf_replaced: int = 0
    coerced_to_nan: int = 0
    duplicate_rows_found: int = 0


def clean_dataset(
    df: pd.DataFrame,
    features: list[str],
    classes: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a tidy frame with columns: features, label, score, group, name, orig_index."""
    assert_features_allowed(features)
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Selected features not found in the file: {missing_cols}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' is missing.")

    log: list[dict] = []
    notes: list[str] = []
    n = len(df)
    log.append({"stage": "Rows loaded", "removed": 0, "remaining": n, "note": ""})

    work = pd.DataFrame(index=df.index)
    for f in features:
        raw = df[f]
        numeric = pd.to_numeric(raw, errors="coerce")
        newly_nan = int((numeric.isna() & raw.notna()).sum())
        work[f] = numeric
    coerced = int(sum(
        (pd.to_numeric(df[f], errors="coerce").isna() & df[f].notna()).sum() for f in features
    ))

    work["label"] = normalise_labels(df[TARGET_COLUMN])
    work["score"] = (
        pd.to_numeric(df[SCORE_COLUMN], errors="coerce") if SCORE_COLUMN in df.columns else np.nan
    )
    bad_score = work["score"].notna() & ((work["score"] < 0) | (work["score"] > 1))
    if bad_score.any():
        work.loc[bad_score, "score"] = np.nan
        notes.append(f"{int(bad_score.sum())} koi_score values outside [0,1] were set to missing.")
    work["group"] = df[GROUP_COLUMN] if GROUP_COLUMN in df.columns else np.nan
    work["name"] = df[NAME_COLUMN] if NAME_COLUMN in df.columns else np.nan
    work["orig_index"] = df.index

    missing_before = work[features].isna().sum()

    # 2. missing / unsupported labels ---------------------------------------------------
    keep = work["label"].notna()
    removed = int((~keep).sum())
    work = work[keep]
    log.append({"stage": "Missing/unsupported disposition", "removed": removed,
                "remaining": len(work),
                "note": "Only FALSE POSITIVE, CANDIDATE, CONFIRMED are supported."})

    # 3. duplicates -----------------------------------------------------------------------
    key_cols = ["name"] if work["name"].notna().any() else features + ["label", "group"]
    dup_mask = work.duplicated(subset=key_cols, keep="first")
    full_dups = int(work.duplicated(subset=features + ["label", "score", "group"], keep=False).sum())
    removed = int(dup_mask.sum())
    work = work[~dup_mask]
    log.append({"stage": "Duplicate records", "removed": removed, "remaining": len(work),
                "note": f"Duplicates identified by {'kepoi_name' if key_cols == ['name'] else 'feature values + label'}."})

    # 4. infinities ------------------------------------------------------------------------
    inf_mask = np.isinf(work[features].to_numpy(dtype=float))
    inf_count = int(inf_mask.sum())
    if inf_count:
        work[features] = work[features].replace([np.inf, -np.inf], np.nan)
    log.append({"stage": "Infinite values -> NaN", "removed": 0, "remaining": len(work),
                "note": f"{inf_count} infinite cell values replaced with NaN (no rows removed)."})
    log.append({"stage": "Non-numeric values -> NaN", "removed": 0, "remaining": len(work),
                "note": f"{coerced} non-numeric cell values coerced to NaN (no rows removed)."})

    # 5. all-features-missing --------------------------------------------------------------
    all_missing = work[features].isna().all(axis=1)
    removed = int(all_missing.sum())
    work = work[~all_missing]
    log.append({"stage": "All selected features missing", "removed": removed,
                "remaining": len(work), "note": "Such rows cannot be imputed meaningfully."})

    # 6. class restriction -----------------------------------------------------------------
    if classes is not None:
        classes = [c for c in CLASS_ORDER if c in classes]
        keep = work["label"].isin(classes)
        removed = int((~keep).sum())
        work = work[keep]
        log.append({"stage": "Restricted to selected classes", "removed": removed,
                    "remaining": len(work), "note": ", ".join(classes)})

    present = [c for c in CLASS_ORDER if c in set(work["label"])]
    if len(present) < 2:
        raise ValueError(f"After cleaning, fewer than two classes remain ({present}).")

    work = work.reset_index(drop=True)
    report = CleaningReport(
        removal_log=pd.DataFrame(log), notes=notes, missing_before=missing_before,
        inf_replaced=inf_count, coerced_to_nan=coerced, duplicate_rows_found=full_dups,
    )
    return work, report


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    mode_requested: str
    mode_used: str
    warnings: list[str] = field(default_factory=list)

    def summary(self, labels: pd.Series) -> pd.DataFrame:
        rows = []
        total = len(labels)
        for name, idx in (("train", self.train_idx), ("validation", self.val_idx), ("test", self.test_idx)):
            counts = labels.iloc[idx].value_counts()
            row = {"split": name, "rows": len(idx), "share": len(idx) / total}
            for c in CLASS_ORDER:
                if c in set(labels):
                    row[c] = int(counts.get(c, 0))
            rows.append(row)
        return pd.DataFrame(rows)


def _each_class_present(labels: np.ndarray, idx: np.ndarray, classes: list[str], minimum: int = 2) -> bool:
    sub = labels[idx]
    return all((sub == c).sum() >= minimum for c in classes)


def _grouped_split(clean, seed, test_size, val_size):
    labels = clean["label"].to_numpy()
    classes = sorted(set(labels))
    # rows without a kepid become singleton groups
    groups = clean["group"].copy()
    missing = groups.isna()
    groups = groups.astype("object")
    groups[missing] = [f"_solo_{i}" for i in np.flatnonzero(missing)]
    groups = groups.astype(str).to_numpy()
    n_groups = len(set(groups))
    if n_groups < 30:
        return None, f"Only {n_groups} distinct host stars; too few for a grouped split."

    n_test_splits = max(2, round(1 / test_size))
    n_val_splits = max(2, round(1 / (val_size / (1 - test_size))))
    all_idx = np.arange(len(clean))

    for attempt in range(10):
        s = seed + attempt
        try:
            outer = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=s)
            rest_idx, test_idx = next(outer.split(all_idx, labels, groups))
            inner = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=s)
            tr_rel, val_rel = next(inner.split(rest_idx, labels[rest_idx], groups[rest_idx]))
        except ValueError as exc:
            return None, f"Grouped split failed: {exc}"
        train_idx, val_idx = rest_idx[tr_rel], rest_idx[val_rel]
        if all(_each_class_present(labels, idx, classes) for idx in (train_idx, val_idx, test_idx)):
            # hard guarantee: no host star in two splits
            g_train, g_val, g_test = (set(groups[i]) for i in (train_idx, val_idx, test_idx))
            if not (g_train & g_val or g_train & g_test or g_val & g_test):
                return (np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)), None
    return None, "Could not find a grouped split where every split contains every class."


def _stratified_split(clean, seed, test_size, val_size):
    labels = clean["label"].to_numpy()
    counts = pd.Series(labels).value_counts()
    if counts.min() < 3:
        raise ValueError(
            f"Class '{counts.idxmin()}' has only {counts.min()} rows after cleaning; "
            "a stratified three-way split is impossible."
        )
    idx = np.arange(len(clean))
    rest, test = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(rest, test_size=rel_val, stratify=labels[rest], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def split_dataset(
    clean: pd.DataFrame,
    mode: str = "grouped",
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SplitResult:
    """Split into ~70/15/15. ``mode`` is 'grouped' (by kepid) or 'stratified'."""
    if mode not in {"grouped", "stratified"}:
        raise ValueError("mode must be 'grouped' or 'stratified'")
    warnings: list[str] = []
    used = mode
    result = None
    if mode == "grouped":
        if "group" not in clean.columns or clean["group"].isna().all():
            warnings.append(
                "Host-star identifier (kepid) is unavailable, so a stratified row-level "
                "split was used instead. Objects around the same star may appear in "
                "different splits."
            )
            used = "stratified"
        else:
            result, reason = _grouped_split(clean, seed, test_size, val_size)
            if result is None:
                warnings.append(f"Grouped split not possible ({reason}) Falling back to a stratified row split.")
                used = "stratified"
    if result is None:
        result = _stratified_split(clean, seed, test_size, val_size)
        used = "stratified"
    train_idx, val_idx, test_idx = result
    return SplitResult(train_idx, val_idx, test_idx, mode, used, warnings)


# --------------------------------------------------------------------------------------
# Feature preprocessing (fit on train only)
# --------------------------------------------------------------------------------------


class FeaturePreprocessor:
    """Optional log10 -> median imputation -> StandardScaler.

    ``fit`` must be called with TRAINING data only. The log10 step is a fixed
    deterministic transform (not learned), so it cannot leak information.
    """

    def __init__(self, features: list[str], log_features: Optional[list[str]] = None):
        self.features = list(features)
        self.log_features = [f for f in (log_features or []) if f in self.features]
        self.kept_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.medians_: dict[str, float] = {}

    def _to_matrix(self, X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        data = X[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()
        for col in self.log_features:
            if col in data.columns:
                values = data[col].astype(float)
                data[col] = np.where(values > 0, np.log10(values.where(values > 0, 1.0)), np.nan)
        return data.to_numpy(dtype=float)

    def fit(self, X_train: pd.DataFrame) -> "FeaturePreprocessor":
        matrix = self._to_matrix(X_train, self.features)
        all_nan = np.isnan(matrix).all(axis=0)
        self.dropped_features_ = [f for f, bad in zip(self.features, all_nan) if bad]
        self.kept_features_ = [f for f, bad in zip(self.features, all_nan) if not bad]
        if not self.kept_features_:
            raise ValueError("Every selected feature is entirely missing in the training set.")
        matrix = self._to_matrix(X_train, self.kept_features_)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(matrix)
        self.scaler_ = StandardScaler().fit(imputed)
        self.medians_ = dict(zip(self.kept_features_, self.imputer_.statistics_))
        return self

    def transform_imputed(self, X: pd.DataFrame) -> np.ndarray:
        """Imputed but NOT scaled (used by tree models)."""
        return self.imputer_.transform(self._to_matrix(X, self.kept_features_)).astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Imputed and standardised (used by ANN and logistic regression)."""
        imputed = self.imputer_.transform(self._to_matrix(X, self.kept_features_))
        return self.scaler_.transform(imputed).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FeaturePreprocessor":
        return joblib.load(path)


# --------------------------------------------------------------------------------------
# One-call preparation used by the app, experiments and tests
# --------------------------------------------------------------------------------------


@dataclass
class PreparedData:
    clean: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    class_names: list[str]
    preprocessor: FeaturePreprocessor
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_train_imp: np.ndarray
    X_val_imp: np.ndarray
    X_test_imp: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cleaning: CleaningReport
    split: SplitResult
    seed: int
    log_features: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def planet_like_indices(self) -> list[int]:
        from .data_validation import PLANET_LIKE_CLASSES
        return [i for i, c in enumerate(self.class_names) if c in PLANET_LIKE_CLASSES]

    @property
    def positive_index(self) -> int:
        """Positive class for binary metrics: the last class in CLASS_ORDER (CANDIDATE for DR25)."""
        return self.n_classes - 1

    def split_summary(self) -> pd.DataFrame:
        return self.split.summary(self.clean["label"])


def prepare_data(
    df: pd.DataFrame,
    features: list[str],
    split_mode: str = "grouped",
    seed: int = 42,
    classes: Optional[list[str]] = None,
    log_features: Optional[list[str]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> PreparedData:
    log_features = DEFAULT_LOG_FEATURES if log_features is None else log_features
    clean, report = clean_dataset(df, features, classes)
    split = split_dataset(clean, split_mode, seed, test_size, val_size)

    train = clean.iloc[split.train_idx].reset_index(drop=True)
    val = clean.iloc[split.val_idx].reset_index(drop=True)
    test = clean.iloc[split.test_idx].reset_index(drop=True)

    pre = FeaturePreprocessor(features, log_features).fit(train)  # TRAIN ONLY
    kept = pre.kept_features_
    warnings = list(split.warnings)
    if pre.dropped_features_:
        warnings.append(
            "Features entirely missing in the training set were dropped: "
            + ", ".join(pre.dropped_features_)
        )

    class_names = [c for c in CLASS_ORDER if c in set(clean["label"])]
    mapping = {c: i for i, c in enumerate(class_names)}

    return PreparedData(
        clean=clean, train=train, val=val, test=test, features=kept, class_names=class_names,
        preprocessor=pre,
        X_train=pre.transform(train), X_val=pre.transform(val), X_test=pre.transform(test),
        X_train_imp=pre.transform_imputed(train), X_val_imp=pre.transform_imputed(val),
        X_test_imp=pre.transform_imputed(test),
        y_train=train["label"].map(mapping).to_numpy(dtype=int),
        y_val=val["label"].map(mapping).to_numpy(dtype=int),
        y_test=test["label"].map(mapping).to_numpy(dtype=int),
        cleaning=report, split=split, seed=seed,
        log_features=[f for f in log_features if f in kept], warnings=warnings,
    )
