from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from . import ann_model, baseline_models
from .data_validation import PLANET_LIKE_CLASSES
from .evaluation import compute_metrics
from .preprocessing import PreparedData

STRATEGY_NAMES = {
    "A": "A: Baseline (all eligible)",
    "B": "B: High-confidence filtered",
    "C": "C: Confidence-weighted",
}
STRATEGY_SHORT = {"A": "Baseline", "B": "Filtered", "C": "Weighted"}


@dataclass
class StrategyParams:
    threshold: float = 0.80            # Experiment B filter threshold t
    min_weight: float = 0.05           # Experiment C weight floor
    weight_exponent: float = 1.0       # 1.0 = the proposed heuristic (weight = score)
    missing_score_policy: str = "exclude"   # "exclude" or "neutral"
    neutral_weight: float = 0.5
    target_mean_weight: float = 1.0    # weights are rescaled to this training mean
    filter_validation: bool = True     # apply the B filter to the validation cohort too
    hc_eval_threshold: Optional[float] = None  # threshold for the high-confidence test subset

    def validate(self) -> "StrategyParams":
        if not 0.5 <= self.threshold <= 1.0:
            raise ValueError("The confidence threshold must be between 0.5 and 1.0.")
        if not 0.0 <= self.min_weight < 1.0:
            raise ValueError("min_weight must be in [0, 1).")
        if self.weight_exponent <= 0:
            raise ValueError("weight_exponent must be positive.")
        if self.missing_score_policy not in {"exclude", "neutral"}:
            raise ValueError("missing_score_policy must be 'exclude' or 'neutral'.")
        return self

    @property
    def eval_threshold(self) -> float:
        return self.threshold if self.hc_eval_threshold is None else self.hc_eval_threshold


# --------------------------------------------------------------------------------------
# Score-based masks and weights
# --------------------------------------------------------------------------------------


def planet_like_mask(labels: pd.Series) -> np.ndarray:
    return labels.isin(PLANET_LIKE_CLASSES).to_numpy()


def high_confidence_mask(labels: pd.Series, scores: pd.Series, threshold: float) -> np.ndarray:
    """Candidate/Confirmed kept if score >= t; False Positive kept if score <= 1 - t.

    Rows with a missing score are never retained.
    """
    s = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    planet = planet_like_mask(labels)
    fp = ~planet
    with np.errstate(invalid="ignore"):
        keep = (planet & (s >= threshold - 1e-12)) | (fp & (s <= (1.0 - threshold) + 1e-12))
    return keep & ~np.isnan(s)


def confidence_weights(labels: pd.Series, scores: pd.Series, params: StrategyParams) -> np.ndarray:
    """Raw (un-normalised) weights: planet-like -> score, False Positive -> 1 - score.

    Missing scores give NaN (to be excluded) or a neutral weight, per the policy.
    A floor ``min_weight`` is applied after the optional exponent.
    """
    params.validate()
    s = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    planet = planet_like_mask(labels)
    base = np.where(planet, s, 1.0 - s)
    w = np.power(np.clip(base, 0.0, 1.0), params.weight_exponent)
    w = np.maximum(w, params.min_weight)
    missing = np.isnan(s)
    if params.missing_score_policy == "neutral":
        w = np.where(missing, params.neutral_weight, w)
    else:
        w = np.where(missing, np.nan, w)
    return w


# --------------------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------------------


@dataclass
class Cohort:
    strategy: str
    X_train: np.ndarray
    X_train_imp: np.ndarray
    y_train: np.ndarray
    w_train: Optional[np.ndarray]
    X_val: np.ndarray
    X_val_imp: np.ndarray
    y_val: np.ndarray
    w_val: Optional[np.ndarray]
    info: dict = field(default_factory=dict)


def build_cohort(data: PreparedData, strategy: str, params: StrategyParams) -> Cohort:
    params.validate()
    train, val = data.train, data.val
    names = data.class_names
    n_before = len(train)
    keep_tr = np.ones(len(train), dtype=bool)
    keep_va = np.ones(len(val), dtype=bool)
    w_tr = w_va = None
    notes: list[str] = []

    if strategy in {"B", "C"} and train["score"].notna().sum() == 0:
        raise ValueError("koi_score is not available, so Experiments B and C cannot run.")

    if strategy == "B":
        keep_tr = high_confidence_mask(train["label"], train["score"], params.threshold)
        if params.filter_validation:
            keep_va = high_confidence_mask(val["label"], val["score"], params.threshold)
        notes.append(
            "This model is trained on a SMALLER training population than the baseline, so any "
            "difference reflects both label cleanliness and reduced data."
        )
    elif strategy == "C":
        raw_tr = confidence_weights(train["label"], train["score"], params)
        raw_va = confidence_weights(val["label"], val["score"], params)
        keep_tr, keep_va = ~np.isnan(raw_tr), ~np.isnan(raw_va)
        if keep_tr.sum() == 0:
            raise ValueError("No training rows remain with a usable koi_score.")
        scale = params.target_mean_weight / raw_tr[keep_tr].mean()   # TRAIN mean only
        w_tr, w_va = (raw_tr * scale)[keep_tr], (raw_va * scale)[keep_va]
        notes.append(
            "Weights are a proposed heuristic from the catalogue's vetting score "
            "(planet-like: weight = score; False Positive: weight = 1 - score). They are NOT "
            "calibrated probabilities of planetary truth, and weights apply to training only."
        )
        if params.missing_score_policy == "exclude":
            notes.append(f"{int((~keep_tr).sum())} training rows without a valid score were excluded.")
        else:
            notes.append(f"Rows without a score received a neutral weight of {params.neutral_weight}.")

    y_tr, y_va = data.y_train[keep_tr], data.y_val[keep_va]
    missing_classes = [names[i] for i in range(len(names)) if (y_tr == i).sum() < 2]
    if missing_classes:
        raise ValueError(
            f"Strategy {strategy} leaves fewer than 2 training rows for class(es) {missing_classes}. "
            "Lower the threshold or choose another strategy."
        )
    if (y_va == np.arange(len(names))[:, None]).sum(axis=1).min() < 1:
        raise ValueError(f"Strategy {strategy} leaves a validation class empty; relax the settings.")

    removed_by_class = {
        names[i]: int(((data.y_train == i) & ~keep_tr).sum()) for i in range(len(names))
    }
    info = {
        "strategy": strategy, "strategy_name": STRATEGY_NAMES[strategy],
        "n_train_before": n_before, "n_train": int(keep_tr.sum()),
        "n_train_removed": int((~keep_tr).sum()), "removed_by_class": removed_by_class,
        "train_class_counts": {names[i]: int((y_tr == i).sum()) for i in range(len(names))},
        "n_val": int(keep_va.sum()), "n_val_before": len(val),
        "train_fraction_retained": float(keep_tr.mean()),
        "weights_summary": None if w_tr is None else {
            "mean": float(w_tr.mean()), "min": float(w_tr.min()), "max": float(w_tr.max()),
            "effective_sample_size": float(w_tr.sum() ** 2 / (w_tr ** 2).sum()),
        },
        "notes": notes,
    }
    return Cohort(
        strategy=strategy,
        X_train=data.X_train[keep_tr], X_train_imp=data.X_train_imp[keep_tr], y_train=y_tr, w_train=w_tr,
        X_val=data.X_val[keep_va], X_val_imp=data.X_val_imp[keep_va], y_val=y_va, w_val=w_va,
        info=info,
    )


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass
class ModelResult:
    model_name: str               # e.g. "ANN", "Random Forest"
    family: str                   # "ANN" or "Traditional"
    strategy: str                 # "A" / "B" / "C"
    metrics: dict                 # full original test set
    metrics_hc: Optional[dict]    # high-confidence part of the test set
    proba_test: np.ndarray
    n_train: int
    train_seconds: float
    history: Optional[dict] = None
    ann_summary: Optional[dict] = None
    observations: list = field(default_factory=list)
    model: object = None          # fitted model object (not serialised to CSV)

    @property
    def label(self) -> str:
        return f"{self.model_name} [{STRATEGY_SHORT[self.strategy]}]"


@dataclass
class ExperimentResult:
    strategy: str
    params: StrategyParams
    cohort_info: dict
    models: list[ModelResult]
    ann_config: Optional[dict] = None
    seed: int = 42
    split_mode: str = "grouped"

    def get(self, model_name: str) -> Optional[ModelResult]:
        return next((m for m in self.models if m.model_name == model_name), None)


def test_high_confidence_mask(data: PreparedData, params: StrategyParams) -> Optional[np.ndarray]:
    """Mask over the common test set selecting rows that pass the high-confidence rule."""
    if data.test["score"].notna().sum() == 0:
        return None
    return high_confidence_mask(data.test["label"], data.test["score"], params.eval_threshold)


def _evaluate(data: PreparedData, proba: np.ndarray, hc_mask: Optional[np.ndarray]):
    n = data.n_classes
    full = compute_metrics(data.y_test, proba, n, data.positive_index)
    hc = None
    if hc_mask is not None and hc_mask.sum() > 0:
        hc = compute_metrics(data.y_test[hc_mask], proba[hc_mask], n, data.positive_index)
    return full, hc


def run_strategy(
    data: PreparedData,
    strategy: str,
    params: Optional[StrategyParams] = None,
    ann_config: Optional[ann_model.ANNConfig] = None,
    include_ann: bool = True,
    baselines: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    keep_models: bool = True,
) -> ExperimentResult:
    """Train the ANN and the requested baselines for ONE strategy and evaluate on the common test set."""
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    params = (params or StrategyParams()).validate()
    ann_config = ann_config or ann_model.ANNConfig(seed=data.seed)
    baselines = baseline_models.available_baselines() if baselines is None else baselines

    cohort = build_cohort(data, strategy, params)
    hc_mask = test_high_confidence_mask(data, params)
    results: list[ModelResult] = []

    if include_ann:
        trained = ann_model.train_ann(
            cohort.X_train, cohort.y_train, cohort.X_val, cohort.y_val, data.n_classes, ann_config,
            sample_weight_train=cohort.w_train, sample_weight_val=cohort.w_val,
            progress_callback=progress_callback,
        )
        proba = ann_model.predict_proba(trained.model, data.X_test)
        full, hc = _evaluate(data, proba, hc_mask)
        results.append(ModelResult(
            "ANN", "ANN", strategy, full, hc, proba, cohort.info["n_train"], trained.train_seconds,
            history=trained.history, ann_summary=trained.summary, observations=trained.observations,
            model=trained.model if keep_models else None,
        ))

    for name in baselines:
        start = time.time()
        scaled = baseline_models.uses_scaled_features(name)
        X_tr = cohort.X_train if scaled else cohort.X_train_imp
        X_te = data.X_test if scaled else data.X_test_imp
        model = baseline_models.train_baseline(name, X_tr, cohort.y_train, data.n_classes,
                                               cohort.w_train, data.seed)
        proba = baseline_models.predict_proba_aligned(model, X_te, data.n_classes)
        full, hc = _evaluate(data, proba, hc_mask)
        results.append(ModelResult(
            name, "Traditional", strategy, full, hc, proba, cohort.info["n_train"],
            time.time() - start, model=model if keep_models else None,
        ))

    return ExperimentResult(
        strategy=strategy, params=params, cohort_info=cohort.info, models=results,
        ann_config=ann_config.to_dict() if include_ann else None, seed=data.seed,
        split_mode=data.split.mode_used,
    )


def run_all_strategies(data: PreparedData, strategies=("A", "B", "C"), **kwargs) -> dict[str, ExperimentResult]:
    out = {}
    for s in strategies:
        out[s] = run_strategy(data, s, **kwargs)
    return out


# --------------------------------------------------------------------------------------
# Tabulation and persistence
# --------------------------------------------------------------------------------------


def results_to_frame(results: dict[str, ExperimentResult], subset: str = "full") -> pd.DataFrame:
    """Flatten results into a comparison table. ``subset``: 'full' test set or 'hc' (high-confidence part)."""
    rows = []
    for strat, exp in results.items():
        for m in exp.models:
            metrics = m.metrics if subset == "full" else m.metrics_hc
            if metrics is None:
                continue
            rows.append({
                "model": m.model_name, "family": m.family, "strategy": STRATEGY_NAMES[strat],
                "strategy_key": strat, "n_train": m.n_train, **metrics,
            })
    return pd.DataFrame(rows)


def save_experiment_artifacts(
    results: dict[str, ExperimentResult], data: PreparedData, directory: Path, tag: str = "run",
) -> dict:
    """Persist models, the fitted preprocessor, feature names and configs together."""
    directory = Path(directory) / tag
    directory.mkdir(parents=True, exist_ok=True)
    data.preprocessor.save(directory / "preprocessor.joblib")
    meta = {
        "features": data.features, "log_features": data.log_features, "class_names": data.class_names,
        "seed": data.seed, "split_mode_used": data.split.mode_used,
        "split_sizes": {"train": len(data.train), "validation": len(data.val), "test": len(data.test)},
        "experiments": {},
    }
    saved: dict[str, list[str]] = {}
    for strat, exp in results.items():
        meta["experiments"][strat] = {
            "params": asdict(exp.params), "cohort_info": exp.cohort_info, "ann_config": exp.ann_config,
        }
        saved[strat] = []
        for m in exp.models:
            if m.model is None:
                continue
            stem = f"{strat}_{m.model_name.replace(' ', '_')}"
            if m.family == "ANN":
                path = ann_model.save_ann(
                    ann_model.ANNTrainingResult(
                        model=m.model, config=ann_model.ANNConfig(**{**exp.ann_config, "hidden_units": tuple(exp.ann_config["hidden_units"])}),
                        history=m.history or {}, best_epoch=(m.ann_summary or {}).get("best_epoch", 0),
                        epochs_run=(m.ann_summary or {}).get("epochs_run", 0), n_features=len(data.features),
                        n_classes=data.n_classes, n_train=m.n_train, n_val=0, weighted=strat == "C",
                        train_seconds=m.train_seconds, summary=m.ann_summary or {}, observations=m.observations,
                    ), directory, stem)
            else:
                path = directory / f"{stem}.joblib"
                joblib.dump(m.model, path)
            saved[strat].append(str(path))
    (directory / "experiment_config.json").write_text(json.dumps(meta, indent=2, default=float))
    return {"directory": str(directory), "files": saved}
