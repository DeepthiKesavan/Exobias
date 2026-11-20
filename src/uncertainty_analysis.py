from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from . import baseline_models
from .ann_model import ANNConfig
from .experiments import STRATEGY_NAMES, StrategyParams, results_to_frame, run_strategy
from .preprocessing import PreparedData, prepare_data

DEFAULT_BASELINES = ["Logistic Regression", "Random Forest"]
KEY_METRICS = ["accuracy", "f1_macro", "log_loss", "brier", "ece", "roc_auc"]

ProgressFn = Optional[Callable[[int, int, str], None]]


def _run_and_tabulate(data: PreparedData, strategies, params: StrategyParams, ann_config: Optional[ANNConfig],
                      include_ann: bool, baselines: list[str], subset: str = "full", **extra) -> pd.DataFrame:
    frames = []
    for s in strategies:
        res = run_strategy(data, s, params=params, ann_config=ann_config, include_ann=include_ann,
                           baselines=baselines, keep_models=False)
        frame = results_to_frame({s: res}, subset=subset)
        for k, v in extra.items():
            frame[k] = v
        frame["cohort_removed"] = res.cohort_info["n_train_removed"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _strategies_available(data: PreparedData, wanted=("A", "B", "C")) -> list[str]:
    has_score = data.train["score"].notna().any()
    return [s for s in wanted if s == "A" or has_score]


# --------------------------------------------------------------------------------------
def multi_seed_experiment(
    df: pd.DataFrame, features: list[str], seeds: list[int], split_mode: str = "grouped",
    params: Optional[StrategyParams] = None, ann_config: Optional[ANNConfig] = None,
    include_ann: bool = False, baselines: Optional[list[str]] = None, strategies=("A", "B", "C"),
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Repeat the whole pipeline (split + training) for each seed. Long-format results."""
    params = params or StrategyParams()
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    frames = []
    for i, seed in enumerate(seeds):
        data = prepare_data(df, features, split_mode=split_mode, seed=seed)
        cfg = replace(ann_config, seed=seed) if ann_config is not None else ANNConfig(seed=seed)
        frames.append(_run_and_tabulate(data, _strategies_available(data, strategies), params, cfg,
                                        include_ann, baselines, seed=seed, split_used=data.split.mode_used))
        if progress:
            progress(i + 1, len(seeds), f"seed {seed} done")
    return pd.concat(frames, ignore_index=True)


def summarise_across_seeds(table: pd.DataFrame, metrics=KEY_METRICS) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric per (model, strategy)."""
    grouped = table.groupby(["model", "strategy"], sort=False)
    out = grouped.agg(n_runs=("seed", "nunique"), n_train_mean=("n_train", "mean"))
    for m in metrics:
        out[f"{m}_mean"] = grouped[m].mean()
        out[f"{m}_std"] = grouped[m].std(ddof=1)
    return out.reset_index()


STABILITY_PAIRS = [
    ("ANN", "B", "ANN", "A", "ANN filtered vs ANN baseline"),
    ("ANN", "C", "ANN", "A", "ANN weighted vs ANN baseline"),
    ("Random Forest", "B", "Random Forest", "A", "Random Forest filtered vs baseline"),
    ("Random Forest", "C", "Random Forest", "A", "Random Forest weighted vs baseline"),
    ("Logistic Regression", "B", "Logistic Regression", "A", "Logistic Regression filtered vs baseline"),
    ("Logistic Regression", "C", "Logistic Regression", "A", "Logistic Regression weighted vs baseline"),
    ("ANN", "A", "Random Forest", "A", "ANN vs Random Forest (baseline cohort)"),
    ("ANN", "A", "Logistic Regression", "A", "ANN vs Logistic Regression (baseline cohort)"),
]


def stability_assessment(table: pd.DataFrame, metric: str = "f1_macro", confidence: float = 0.95,
                         pairs=STABILITY_PAIRS) -> pd.DataFrame:
    """Paired per-seed differences (first minus second) and a plain-language verdict."""
    rows = []
    for m1, s1, m2, s2, label in pairs:
        a = table[(table.model == m1) & (table.strategy_key == s1)].set_index("seed")[metric]
        b = table[(table.model == m2) & (table.strategy_key == s2)].set_index("seed")[metric]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        diff = (a.loc[common] - b.loc[common]).to_numpy(float)
        n = len(diff)
        mean = float(diff.mean())
        if n >= 3 and diff.std(ddof=1) > 0:
            sem = diff.std(ddof=1) / np.sqrt(n)
            half = stats.t.ppf((1 + confidence) / 2, n - 1) * sem
            lo, hi = mean - half, mean + half
        else:
            lo = hi = np.nan
        wins = int((diff > 0).sum())
        if n < 3:
            verdict = "Insufficient seeds (need at least 3)"
        elif np.isnan(lo):
            verdict = "Identical in every seed"
        elif lo > 0:
            verdict = "Stable improvement"
        elif hi < 0:
            verdict = "Stable decline"
        else:
            verdict = "No reliable difference"
        rows.append({"comparison": label, "metric": metric, "n_seeds": n, "mean_diff": mean,
                     "ci_low": lo, "ci_high": hi, "first_better_in_seeds": f"{wins}/{n}", "verdict": verdict})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
def threshold_sweep(
    data: PreparedData, thresholds=(0.70, 0.80, 0.90), params: Optional[StrategyParams] = None,
    ann_config: Optional[ANNConfig] = None, include_ann: bool = False,
    baselines: Optional[list[str]] = None, reference_hc_threshold: float = 0.80,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Experiment B at several thresholds. All models are scored on the SAME high-confidence
    test subset (defined by ``reference_hc_threshold``) so that sub-set results are comparable."""
    params = params or StrategyParams()
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    frames = []
    base = _run_and_tabulate(data, ["A"], params, ann_config, include_ann, baselines, threshold=np.nan,
                             subset="full")
    base_hc = _run_and_tabulate(data, ["A"], replace(params, hc_eval_threshold=reference_hc_threshold), ann_config,
                                include_ann, baselines, threshold=np.nan, subset="hc")
    base["test_subset"], base_hc["test_subset"] = "full test set", f"high-confidence test (t={reference_hc_threshold})"
    frames += [base, base_hc]
    for i, t in enumerate(thresholds):
        p = replace(params, threshold=t, hc_eval_threshold=reference_hc_threshold)
        full = _run_and_tabulate(data, ["B"], p, ann_config, include_ann, baselines, threshold=t, subset="full")
        hc = _run_and_tabulate(data, ["B"], p, ann_config, include_ann, baselines, threshold=t, subset="hc")
        full["test_subset"], hc["test_subset"] = "full test set", f"high-confidence test (t={reference_hc_threshold})"
        frames += [full, hc]
        if progress:
            progress(i + 1, len(thresholds), f"threshold {t:.2f} done")
    out = pd.concat(frames, ignore_index=True)
    n_total = len(data.train)
    out["train_fraction_of_baseline"] = out["n_train"] / n_total
    return out


def weight_sensitivity(
    data: PreparedData, settings: Optional[list[dict]] = None, params: Optional[StrategyParams] = None,
    ann_config: Optional[ANNConfig] = None, include_ann: bool = False,
    baselines: Optional[list[str]] = None, progress: ProgressFn = None,
) -> pd.DataFrame:
    """Experiment C under several weighting settings (exponent, floor, missing-score policy)."""
    params = params or StrategyParams()
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    settings = settings or [
        {"label": "score (default heuristic)", "weight_exponent": 1.0, "min_weight": 0.05},
        {"label": "score, floor 0.20", "weight_exponent": 1.0, "min_weight": 0.20},
        {"label": "score^0.5 (softer)", "weight_exponent": 0.5, "min_weight": 0.05},
        {"label": "score^2 (sharper)", "weight_exponent": 2.0, "min_weight": 0.05},
        {"label": "missing score -> neutral 0.5", "weight_exponent": 1.0, "min_weight": 0.05,
         "missing_score_policy": "neutral"},
    ]
    frames = []
    ref = _run_and_tabulate(data, ["A"], params, ann_config, include_ann, baselines, weight_setting="A: unweighted baseline")
    frames.append(ref)
    for i, s in enumerate(settings):
        s = dict(s)
        label = s.pop("label")
        frames.append(_run_and_tabulate(data, ["C"], replace(params, **s), ann_config, include_ann, baselines,
                                        weight_setting=label))
        if progress:
            progress(i + 1, len(settings), f"{label} done")
    return pd.concat(frames, ignore_index=True)


def split_comparison(
    df: pd.DataFrame, features: list[str], seeds: list[int], params: Optional[StrategyParams] = None,
    ann_config: Optional[ANNConfig] = None, include_ann: bool = False, baselines: Optional[list[str]] = None,
    strategies=("A", "B", "C"), progress: ProgressFn = None,
) -> pd.DataFrame:
    """Host-star-grouped split versus stratified row split, repeated over seeds."""
    frames = []
    for j, mode in enumerate(("grouped", "stratified")):
        t = multi_seed_experiment(df, features, seeds, mode, params, ann_config, include_ann, baselines, strategies)
        t["split_requested"] = mode
        frames.append(t)
        if progress:
            progress(j + 1, 2, f"{mode} split done")
    return pd.concat(frames, ignore_index=True)


def feature_sensitivity(
    df: pd.DataFrame, features: list[str], seed: int = 42, split_mode: str = "grouped",
    strategy: str = "A", params: Optional[StrategyParams] = None, ann_config: Optional[ANNConfig] = None,
    include_ann: bool = False, baselines: Optional[list[str]] = None, progress: ProgressFn = None,
) -> pd.DataFrame:
    """Drop one feature at a time (and planetary-only / stellar-only sets); report change vs all features."""
    from .data_validation import PLANETARY_FEATURES, STELLAR_FEATURES
    params = params or StrategyParams()
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    scenarios = [("all features", list(features))]
    scenarios += [(f"without {f}", [x for x in features if x != f]) for f in features if len(features) > 2]
    planetary = [f for f in features if f in PLANETARY_FEATURES]
    stellar = [f for f in features if f in STELLAR_FEATURES]
    if len(planetary) >= 2:
        scenarios.append(("planetary features only", planetary))
    if len(stellar) >= 2:
        scenarios.append(("stellar features only", stellar))
    frames = []
    for i, (label, feats) in enumerate(scenarios):
        data = prepare_data(df, feats, split_mode=split_mode, seed=seed)
        cfg = replace(ann_config, seed=seed) if ann_config is not None else ANNConfig(seed=seed)
        if strategy != "A" and not data.train["score"].notna().any():
            continue
        frames.append(_run_and_tabulate(data, [strategy], params, cfg, include_ann, baselines,
                                        feature_set=label, n_features=len(data.features)))
        if progress:
            progress(i + 1, len(scenarios), f"{label} done")
    out = pd.concat(frames, ignore_index=True)
    ref = out[out.feature_set == "all features"].set_index("model")["f1_macro"]
    out["f1_macro_change_vs_all"] = out["f1_macro"] - out["model"].map(ref)
    return out


def class_distribution_comparison(data: PreparedData, params: Optional[StrategyParams] = None) -> pd.DataFrame:
    """Class counts per data split and per strategy training cohort."""
    from .experiments import build_cohort
    params = params or StrategyParams()
    rows = []
    for name, y in (("train", data.y_train), ("validation", data.y_val), ("test", data.y_test)):
        row = {"population": f"{name} split", "rows": len(y)}
        row.update({c: int((y == i).sum()) for i, c in enumerate(data.class_names)})
        rows.append(row)
    for s in _strategies_available(data):
        info = build_cohort(data, s, params).info
        rows.append({"population": f"training cohort {STRATEGY_NAMES[s]}", "rows": info["n_train"],
                     **info["train_class_counts"]})
    out = pd.DataFrame(rows)
    pos = data.class_names[-1]
    out[f"share_{pos}"] = out[pos] / out["rows"]
    return out


def dataset_sensitivity(
    datasets: dict[str, pd.DataFrame], features: list[str], seed: int = 42, split_mode: str = "grouped",
    params: Optional[StrategyParams] = None, ann_config: Optional[ANNConfig] = None,
    include_ann: bool = False, baselines: Optional[list[str]] = None, strategies=("A", "B", "C"),
) -> pd.DataFrame:
    """Same pipeline on several uploaded catalogue versions/sources (needs >= 2 datasets)."""
    params = params or StrategyParams()
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    frames = []
    for name, df in datasets.items():
        usable = [f for f in features if f in df.columns]
        data = prepare_data(df, usable, split_mode=split_mode, seed=seed)
        cfg = replace(ann_config, seed=seed) if ann_config is not None else ANNConfig(seed=seed)
        frames.append(_run_and_tabulate(data, _strategies_available(data, strategies), params, cfg, include_ann,
                                        baselines, dataset=name, n_rows_clean=len(data.clean),
                                        classes=", ".join(data.class_names)))
    return pd.concat(frames, ignore_index=True)


def population_stability(
    df: pd.DataFrame, features: list[str], thresholds=(0.70, 0.80, 0.90), column: str = "koi_prad",
    n_boot: int = 200, seed: int = 0,
) -> pd.DataFrame:
    """How the observed-catalogue median of ``column`` changes with the filter threshold / weights."""
    from .population_analysis import ALL_KEY, CW_KEY, HC_KEY, strategy_weight_vectors, summarize_distribution
    from .preprocessing import clean_dataset
    clean, _ = clean_dataset(df, features)
    rows = []
    allw = np.ones(len(clean))
    s = summarize_distribution(clean[column], allw, n_boot, seed)
    rows.append({"setting": ALL_KEY, "threshold": np.nan, **{k: s.get(k) for k in ("n_rows", "median", "median_ci_low", "median_ci_high", "iqr")}})
    if clean["score"].notna().any():
        for t in thresholds:
            w = strategy_weight_vectors(clean, StrategyParams(threshold=t))
            for key in (HC_KEY, CW_KEY):
                if key == CW_KEY and t != thresholds[0]:
                    continue
                s = summarize_distribution(clean[column], w[key], n_boot, seed)
                rows.append({"setting": f"{key}" + (f" (t={t:.2f})" if key == HC_KEY else ""),
                             "threshold": t if key == HC_KEY else np.nan,
                             **{k: s.get(k) for k in ("n_rows", "median", "median_ci_low", "median_ci_high", "iqr")}})
    out = pd.DataFrame(rows)
    out["median_change_vs_all"] = out["median"] - out.loc[0, "median"]
    out["column"] = column
    return out
