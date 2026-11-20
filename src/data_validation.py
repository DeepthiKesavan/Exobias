from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TARGET_COLUMN = "koi_disposition"
SCORE_COLUMN = "koi_score"
GROUP_COLUMN = "kepid"
NAME_COLUMN = "kepoi_name"

# Order defines the integer encoding: FALSE POSITIVE=0, CANDIDATE=1, CONFIRMED=2.
# For the DR25 binary task the *last* class (CANDIDATE) is the positive class.
CLASS_ORDER = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
PLANET_LIKE_CLASSES = {"CANDIDATE", "CONFIRMED"}

PLANETARY_FEATURES = ["koi_period", "koi_prad", "koi_teq", "koi_insol"]
STELLAR_FEATURES = ["koi_steff", "koi_srad", "koi_smass"]
DEFAULT_FEATURES = PLANETARY_FEATURES + STELLAR_FEATURES

# Features that span many orders of magnitude; log10-transformed by default.
DEFAULT_LOG_FEATURES = ["koi_period", "koi_prad", "koi_insol"]

FORBIDDEN_EXACT = {
    TARGET_COLUMN, SCORE_COLUMN, GROUP_COLUMN, NAME_COLUMN, "kepler_name",
    "koi_pdisposition", "koi_vet_stat", "koi_vet_date", "koi_disp_prov",
    "koi_comment", "rowid", "kepoi_name", "label", "score", "group",
}
FORBIDDEN_PREFIXES = ("koi_fpflag_", "koi_score", "koi_disp", "koi_pdisp")


def is_forbidden_feature(name: str) -> bool:
    return name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIXES)


def assert_features_allowed(features: Iterable[str]) -> None:
    """Raise if any requested feature would leak the label or vetting outcome."""
    bad = [f for f in features if is_forbidden_feature(f)]
    if bad:
        raise ValueError(
            "These columns cannot be used as model inputs because they are the "
            f"target, the Robovetter score, an identifier or a vetting flag: {bad}."
        )


def normalise_label(value) -> Optional[str]:
    """Map raw disposition text to a canonical class name, or None if unsupported."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text if text in CLASS_ORDER else None


def normalise_labels(series: pd.Series) -> pd.Series:
    return series.map(normalise_label)


def class_counts(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.Series:
    """Counts of supported classes plus an 'UNSUPPORTED / MISSING' bucket."""
    if target not in df.columns:
        return pd.Series(dtype=int)
    labels = normalise_labels(df[target])
    counts = labels.value_counts()
    counts = counts.reindex([c for c in CLASS_ORDER if c in counts.index])
    unsupported = int(labels.isna().sum())
    if unsupported:
        counts["UNSUPPORTED / MISSING"] = unsupported
    return counts.astype(int)


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric-looking columns that are allowed as inputs (for the feature picker)."""
    out = []
    for col in df.columns:
        if is_forbidden_feature(col):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            out.append(col)
    return out


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    supported_classes: list[str] = field(default_factory=list)
    available_features: list[str] = field(default_factory=list)
    missing_default_features: list[str] = field(default_factory=list)
    has_score: bool = False
    has_group: bool = False
    has_confirmed: bool = False
    score_missing_fraction: float = float("nan")
    n_rows: int = 0
    n_columns: int = 0

    @property
    def is_trainable(self) -> bool:
        return not self.errors

    @property
    def supports_filtering_and_weighting(self) -> bool:
        return self.has_score

    @property
    def supports_three_class(self) -> bool:
        return all(c in self.supported_classes for c in CLASS_ORDER)

    def as_messages(self) -> list[tuple[str, str]]:
        return (
            [("error", m) for m in self.errors]
            + [("warning", m) for m in self.warnings]
            + [("info", m) for m in self.infos]
        )


def validate_koi_dataframe(df: pd.DataFrame, min_per_class: int = 20) -> ValidationReport:
    rep = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    # ---- target column and classes -------------------------------------------------
    if TARGET_COLUMN not in df.columns:
        rep.errors.append(
            f"Required column '{TARGET_COLUMN}' is missing. Download the KOI table "
            "from the NASA Exoplanet Archive with all columns."
        )
    else:
        counts = class_counts(df)
        rep.class_counts = counts.to_dict()
        rep.supported_classes = [c for c in CLASS_ORDER if c in counts.index]
        rep.has_confirmed = "CONFIRMED" in rep.supported_classes
        unsupported = int(counts.get("UNSUPPORTED / MISSING", 0))
        if unsupported:
            rep.warnings.append(
                f"{unsupported} rows have a missing or unsupported disposition and "
                "will be removed during preprocessing."
            )
        if len(rep.supported_classes) < 2:
            rep.errors.append(
                "Fewer than two supported classes found "
                f"(found: {rep.supported_classes or 'none'}). Classification needs "
                "at least two of FALSE POSITIVE, CANDIDATE, CONFIRMED."
            )
        for cls in rep.supported_classes:
            if counts[cls] < min_per_class:
                rep.errors.append(
                    f"Class '{cls}' has only {counts[cls]} rows (minimum "
                    f"{min_per_class}). Splitting and evaluation would be unreliable."
                )
        if rep.supported_classes:
            rep.infos.append(
                "Classes available for training: " + ", ".join(rep.supported_classes) + "."
            )
        if not rep.has_confirmed:
            rep.infos.append(
                "No CONFIRMED records found. This is expected for the DR25 KOI table, "
                "which mainly uses CANDIDATE and FALSE POSITIVE. The binary experiments "
                "will run; the optional three-class experiment is disabled."
            )

    # ---- features ----------------------------------------------------------------------
    rep.available_features = [f for f in DEFAULT_FEATURES if f in df.columns]
    rep.missing_default_features = [f for f in DEFAULT_FEATURES if f not in df.columns]
    if rep.missing_default_features:
        rep.warnings.append(
            "Default feature columns not present in this file: "
            + ", ".join(rep.missing_default_features) + "."
        )
    # keep only numeric-looking defaults
    numeric_ok = []
    for f in rep.available_features:
        share = pd.to_numeric(df[f], errors="coerce").notna().mean()
        if share < 0.05:
            rep.warnings.append(f"Feature '{f}' is almost entirely non-numeric/missing and was dropped.")
        else:
            numeric_ok.append(f)
            if share < 0.5:
                rep.warnings.append(f"Feature '{f}' is more than 50% missing.")
    rep.available_features = numeric_ok
    if len(rep.available_features) < 2:
        rep.errors.append(
            "Fewer than two usable default feature columns exist in this file. "
            "Pick other numeric columns on the Preprocessing page or upload a full KOI table."
        )

    # ---- score and group columns ---------------------------------------------------------
    rep.has_score = SCORE_COLUMN in df.columns
    if rep.has_score:
        score = pd.to_numeric(df[SCORE_COLUMN], errors="coerce")
        rep.score_missing_fraction = float(score.isna().mean())
        if score.notna().mean() < 0.2:
            rep.has_score = False
            rep.warnings.append(
                f"'{SCORE_COLUMN}' is mostly missing; Experiments B and C are disabled."
            )
        elif rep.score_missing_fraction > 0:
            rep.warnings.append(
                f"{rep.score_missing_fraction:.1%} of '{SCORE_COLUMN}' values are missing. "
                "Those rows are excluded from Experiments B and C by default."
            )
        if score.notna().any() and ((score.dropna() < 0) | (score.dropna() > 1)).any():
            rep.warnings.append(
                f"'{SCORE_COLUMN}' has values outside [0, 1]; they are treated as missing."
            )
    else:
        rep.warnings.append(
            f"Column '{SCORE_COLUMN}' not found: Experiments B and C (filtering / "
            "weighting) are unavailable. Experiment A still works."
        )

    rep.has_group = GROUP_COLUMN in df.columns and df[GROUP_COLUMN].notna().any()
    if not rep.has_group:
        rep.warnings.append(
            f"Column '{GROUP_COLUMN}' not found: host-star grouped splitting is "
            "unavailable and a stratified row-level split will be used instead."
        )

    rep.infos.append(
        "Reminder: koi_score is a Robovetter disposition score from the vetting process. "
        "It is NOT a calibrated probability that an object is a real planet."
    )
    return rep
