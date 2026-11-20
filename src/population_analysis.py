from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .experiments import StrategyParams, confidence_weights, high_confidence_mask

ALL_KEY = "All eligible records"
HC_KEY = "High-confidence filtered"
CW_KEY = "Confidence-weighted"


# --------------------------------------------------------------------------------------
# Strategy weight vectors
# --------------------------------------------------------------------------------------
def strategy_weight_vectors(clean: pd.DataFrame, params: Optional[StrategyParams] = None) -> dict[str, np.ndarray]:
    params = (params or StrategyParams()).validate()
    out = {ALL_KEY: np.ones(len(clean))}
    if clean["score"].notna().sum() > 0:
        out[HC_KEY] = high_confidence_mask(clean["label"], clean["score"], params.threshold).astype(float)
        w = confidence_weights(clean["label"], clean["score"], params)
        out[CW_KEY] = np.nan_to_num(w, nan=0.0)
    return out


# --------------------------------------------------------------------------------------
# Weighted statistics
# --------------------------------------------------------------------------------------
def _valid(values, weights):
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return v[m], w[m]


def weighted_mean(v, w) -> float:
    return float(np.sum(v * w) / np.sum(w))


def weighted_std(v, w) -> float:
    mu = weighted_mean(v, w)
    return float(np.sqrt(np.sum(w * (v - mu) ** 2) / np.sum(w)))


def weighted_quantile(v, w, q):
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return np.interp(q, (cum - 0.5 * w) / cum[-1], v)


def effective_n(w) -> float:
    return float(np.sum(w) ** 2 / np.sum(w ** 2)) if len(w) else 0.0


def _point_stats(v, w) -> dict:
    q1, med, q3 = weighted_quantile(v, w, [0.25, 0.5, 0.75])
    return {"mean": weighted_mean(v, w), "std": weighted_std(v, w), "median": float(med),
            "q25": float(q1), "q75": float(q3), "iqr": float(q3 - q1)}


def summarize_distribution(values, weights, n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> dict:
    """Point summary plus bootstrap CIs for the mean and median."""
    v, w = _valid(values, weights)
    if len(v) < 2:
        return {"n_rows": len(v), "n_effective": float(len(v))}
    out = {"n_rows": int(len(v)), "n_effective": effective_n(w), **_point_stats(v, w)}
    rng = np.random.default_rng(seed)
    means, medians = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(v), len(v))
        vb, wb = v[idx], w[idx]
        means[b] = weighted_mean(vb, wb)
        medians[b] = weighted_quantile(vb, wb, 0.5)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out.update({"mean_ci_low": float(np.percentile(means, lo)), "mean_ci_high": float(np.percentile(means, hi)),
                "median_ci_low": float(np.percentile(medians, lo)), "median_ci_high": float(np.percentile(medians, hi))})
    return out


def summary_table(values, weight_vectors: dict[str, np.ndarray], n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    rows = []
    for name, w in weight_vectors.items():
        rows.append({"strategy": name, **summarize_distribution(values, w, n_boot, seed)})
    return pd.DataFrame(rows)


def weighted_ks(x, wx, y, wy) -> tuple[float, float]:
    """(D, approximate p-value) from weighted ECDFs with Kish effective sample sizes."""
    x, wx = _valid(x, wx)
    y, wy = _valid(y, wy)
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    grid = np.sort(np.concatenate([x, y]))

    def ecdf(v, w):
        order = np.argsort(v)
        v, cw = v[order], np.cumsum(w[order]) / w.sum()
        pos = np.searchsorted(v, grid, side="right")
        return np.where(pos > 0, cw[np.maximum(pos - 1, 0)], 0.0)

    d = float(np.max(np.abs(ecdf(x, wx) - ecdf(y, wy))))
    n1, n2 = effective_n(wx), effective_n(wy)
    n_eff = n1 * n2 / (n1 + n2)
    p = float(stats.kstwobign.sf(np.sqrt(n_eff) * d))
    return d, p


def cohens_d(x, wx, y, wy) -> float:
    x, wx = _valid(x, wx)
    y, wy = _valid(y, wy)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n1, n2 = effective_n(wx), effective_n(wy)
    pooled = np.sqrt(((n1 - 1) * weighted_std(x, wx) ** 2 + (n2 - 1) * weighted_std(y, wy) ** 2) / max(n1 + n2 - 2, 1))
    return float((weighted_mean(x, wx) - weighted_mean(y, wy)) / pooled) if pooled > 0 else float("nan")


def effect_label(d: float) -> str:
    if not np.isfinite(d):
        return "n/a"
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _log10_positive(values):
    v = np.asarray(values, dtype=float)
    out = np.full_like(v, np.nan)
    ok = np.isfinite(v) & (v > 0)
    out[ok] = np.log10(v[ok])
    return out


# --------------------------------------------------------------------------------------
# Comparisons between strategies (nested -> paired bootstrap)
# --------------------------------------------------------------------------------------
def compare_to_reference(values, weight_vectors: dict[str, np.ndarray], reference: str = ALL_KEY,
                         n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    n = len(values)
    ref_w = weight_vectors[reference]
    logv = _log10_positive(values)
    rng = np.random.default_rng(seed)
    others = [k for k in weight_vectors if k != reference]
    boot = {k: {"median": np.empty(n_boot), "mean": np.empty(n_boot)} for k in others}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vb = values[idx]
        vr, wr = _valid(vb, ref_w[idx])
        ref_med, ref_mean = weighted_quantile(vr, wr, 0.5), weighted_mean(vr, wr)
        for k in others:
            vk, wk = _valid(vb, weight_vectors[k][idx])
            boot[k]["median"][b] = weighted_quantile(vk, wk, 0.5) - ref_med
            boot[k]["mean"][b] = weighted_mean(vk, wk) - ref_mean
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    v_ref, w_ref = _valid(values, ref_w)
    rows = []
    for k in others:
        v_k, w_k = _valid(values, weight_vectors[k])
        med_diff = float(weighted_quantile(v_k, w_k, 0.5) - weighted_quantile(v_ref, w_ref, 0.5))
        mean_diff = weighted_mean(v_k, w_k) - weighted_mean(v_ref, w_ref)
        d_ks, p_ks = weighted_ks(values, weight_vectors[k], values, ref_w)
        d_eff = cohens_d(logv, weight_vectors[k], logv, ref_w)
        m_lo, m_hi = np.percentile(boot[k]["median"], [lo, hi])
        a_lo, a_hi = np.percentile(boot[k]["mean"], [lo, hi])
        rows.append({
            "comparison": f"{k}  vs  {reference}", "n_compared": len(v_k), "n_reference": len(v_ref),
            "median_diff": med_diff, "median_diff_ci_low": float(m_lo), "median_diff_ci_high": float(m_hi),
            "median_diff_ci_excludes_0": bool(m_lo > 0 or m_hi < 0),
            "mean_diff": float(mean_diff), "mean_diff_ci_low": float(a_lo), "mean_diff_ci_high": float(a_hi),
            "ks_D": d_ks, "ks_p_approx": p_ks,
            "cohens_d_log10": d_eff, "effect_size_label": effect_label(d_eff),
            "caveat": "Nested samples: read ks_D descriptively; the p-value is approximate and not a valid independent-sample test.",
        })
    return pd.DataFrame(rows)


def compare_label_groups(clean: pd.DataFrame, column: str, weight_vectors: dict[str, np.ndarray],
                         group_a: str = "CANDIDATE", group_b: str = "FALSE POSITIVE",
                         n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Candidate vs False Positive within each strategy (independent groups)."""
    v = clean[column].to_numpy(dtype=float)
    logv = _log10_positive(v)
    a_mask = (clean["label"] == group_a).to_numpy()
    b_mask = (clean["label"] == group_b).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for name, w in weight_vectors.items():
        wa, wb = w * a_mask, w * b_mask
        va, wva = _valid(v, wa)
        vb, wvb = _valid(v, wb)
        if len(va) < 5 or len(vb) < 5:
            continue
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            ia, ib = rng.integers(0, len(va), len(va)), rng.integers(0, len(vb), len(vb))
            diffs[i] = weighted_quantile(va[ia], wva[ia], 0.5) - weighted_quantile(vb[ib], wvb[ib], 0.5)
        d_ks, p_ks = weighted_ks(v, wa, v, wb)
        if np.all(np.isin(w, [0.0, 1.0])):  # unit weights -> exact scipy test
            p_ks = float(stats.ks_2samp(va, vb).pvalue)
        d_eff = cohens_d(logv, wa, logv, wb)
        rows.append({
            "strategy": name, "group_a": group_a, "group_b": group_b,
            "n_a": len(va), "n_b": len(vb),
            "median_a": float(weighted_quantile(va, wva, 0.5)), "median_b": float(weighted_quantile(vb, wvb, 0.5)),
            "median_diff": float(weighted_quantile(va, wva, 0.5) - weighted_quantile(vb, wvb, 0.5)),
            "median_diff_ci_low": float(np.percentile(diffs, 2.5)), "median_diff_ci_high": float(np.percentile(diffs, 97.5)),
            "ks_D": d_ks, "ks_p": p_ks, "cohens_d_log10": d_eff, "effect_size_label": effect_label(d_eff),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Relationships: period-radius and host-star properties
# --------------------------------------------------------------------------------------
def weighted_correlation(x, y, w, method: str = "spearman") -> float:
    """Weighted Pearson on raw values, or 'spearman' = weighted Pearson of (unweighted) ranks."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 5:
        return float("nan")
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
    mx, my = weighted_mean(x, w), weighted_mean(y, w)
    cov = np.sum(w * (x - mx) * (y - my)) / w.sum()
    sx, sy = weighted_std(x, w), weighted_std(y, w)
    return float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def correlation_table(clean: pd.DataFrame, pairs: list[tuple[str, str]], weight_vectors: dict[str, np.ndarray],
                      n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Spearman-style weighted correlations (with bootstrap CI) per strategy for column pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        if a not in clean.columns or b not in clean.columns:
            continue
        xa, xb = clean[a].to_numpy(float), clean[b].to_numpy(float)
        for name, w in weight_vectors.items():
            est = weighted_correlation(xa, xb, w)
            valid = np.flatnonzero(np.isfinite(xa) & np.isfinite(xb) & (w > 0))
            boots = []
            if len(valid) >= 10:
                for _ in range(n_boot):
                    i = valid[rng.integers(0, len(valid), len(valid))]
                    boots.append(weighted_correlation(xa[i], xb[i], w[i]))
            lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
            rows.append({"variable_x": a, "variable_y": b, "strategy": name, "n": len(valid),
                         "spearman_weighted": est, "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)


def binned_median(x, y, w, bins) -> pd.DataFrame:
    """Weighted median of y within bins of x (for host-star-property trends)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    cat = np.digitize(x, bins[1:-1])
    rows = []
    for i in range(len(bins) - 1):
        m = cat == i
        if m.sum() >= 5:
            rows.append({"bin_low": bins[i], "bin_high": bins[i + 1], "bin_mid": 0.5 * (bins[i] + bins[i + 1]),
                         "n": int(m.sum()), "median_y": float(weighted_quantile(y[m], w[m], 0.5))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# ANN / model prediction-based exploratory analysis (test set only)
# --------------------------------------------------------------------------------------
def prediction_based_summary(test: pd.DataFrame, probas: dict[str, np.ndarray], class_names: list[str],
                             column: str = "koi_prad", positive_index: Optional[int] = None,
                             n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """How model predictions reshape a population summary on the held-out test set.

    Exploratory only: predictions measure agreement with catalogue dispositions and are not
    astronomical confirmation. Only held-out rows are used so predictions are not in-sample.
    """
    positive_index = len(class_names) - 1 if positive_index is None else positive_index
    values = test[column].to_numpy(float)
    rows = []
    truth = (test["label"] != "FALSE POSITIVE").to_numpy()
    base = summarize_distribution(values, truth.astype(float), n_boot, seed)
    rows.append({"model": "Catalogue disposition (planet-like)", "predicted_planet_like": int(truth.sum()),
                 "predicted_planet_like_share": float(truth.mean()), "mean_p_planet_like": np.nan,
                 "median_hard": base.get("median", np.nan), "median_hard_ci_low": base.get("median_ci_low", np.nan),
                 "median_hard_ci_high": base.get("median_ci_high", np.nan),
                 "median_prob_weighted": np.nan, "median_prob_weighted_ci_low": np.nan,
                 "median_prob_weighted_ci_high": np.nan})
    for name, proba in probas.items():
        p_pos = np.asarray(proba)[:, positive_index]
        hard = (np.asarray(proba).argmax(axis=1) == positive_index).astype(float)
        hs = summarize_distribution(values, hard, n_boot, seed)
        ps = summarize_distribution(values, p_pos, n_boot, seed)
        rows.append({"model": name, "predicted_planet_like": int(hard.sum()),
                     "predicted_planet_like_share": float(hard.mean()), "mean_p_planet_like": float(p_pos.mean()),
                     "median_hard": hs.get("median", np.nan), "median_hard_ci_low": hs.get("median_ci_low", np.nan),
                     "median_hard_ci_high": hs.get("median_ci_high", np.nan),
                     "median_prob_weighted": ps.get("median", np.nan),
                     "median_prob_weighted_ci_low": ps.get("median_ci_low", np.nan),
                     "median_prob_weighted_ci_high": ps.get("median_ci_high", np.nan)})
    return pd.DataFrame(rows)
