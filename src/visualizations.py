from __future__ import annotations

import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BG, CARD, TEXT, MUTED, GRID = "#070A16", "#101830", "#EEF2FF", "#98A5C8", "#1F2A4A"
PRIMARY, SECONDARY, ACCENT = "#6FD6FF", "#C9D6FF", "#4DF0B0"

STRATEGY_COLORS = {"A": "#3987e5", "B": "#d95926", "C": "#199e70"}
STRATEGY_LABELS = {"A": "A: Baseline", "B": "B: Filtered", "C": "C: Weighted"}
CLASS_COLORS = {"CANDIDATE": "#9085e9", "FALSE POSITIVE": "#c98500", "CONFIRMED": "#199e70"}
MODEL_STYLE = {  # colour, linestyle, marker
    "ANN": ("#9085e9", "-", "o"),
    "Logistic Regression": ("#d55181", "--", "s"),
    "Random Forest": ("#c98500", "-.", "^"),
    "Decision Tree": ("#8da2b5", ":", "D"),
    "XGBoost": ("#199e70", (0, (5, 1)), "v"),
}
POP_COLORS = {"All eligible records": "#3987e5", "High-confidence filtered": "#d95926",
              "Confidence-weighted": "#199e70"}

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro-F1",
    "precision_macro": "Macro precision", "recall_macro": "Macro recall", "log_loss": "Log loss (lower better)",
    "brier": "Brier score (lower better)", "ece": "ECE (lower better)", "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC", "precision_pos": "Precision (planet-like)", "recall_pos": "Recall (planet-like)",
    "f1_pos": "F1 (planet-like)",
}


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": CARD, "legend.edgecolor": GRID, "legend.labelcolor": TEXT,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.dpi": 110, "lines.linewidth": 2.0,
    })


apply_theme()


def fig_to_png(fig, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    return buf.getvalue()


def _finish(fig):
    fig.tight_layout()
    return fig


def _model_style(name: str):
    return MODEL_STYLE.get(name, ("#8da2b5", "-", "o"))


# --------------------------------------------------------------------------------------
# Data exploration
# --------------------------------------------------------------------------------------
def fig_class_distribution(counts: pd.Series, title: str = "Class distribution"):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = list(counts.index)
    colors = [CLASS_COLORS.get(l, MUTED) for l in labels]
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / total:.1%})", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Records")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def fig_missing_values(missing: pd.Series, n_rows: int, title: str = "Missing values per selected feature"):
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * max(len(missing), 3) + 1))
    pct = missing / n_rows * 100
    ax.barh(pct.index[::-1], pct.values[::-1], color=PRIMARY, height=0.55)
    for y, (name, v) in enumerate(zip(pct.index[::-1], pct.values[::-1])):
        ax.text(v, y, f" {v:.1f}% ({int(missing[name]):,})", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, max(pct.max() * 1.35, 1))
    ax.set_xlabel("% of rows missing")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _finish(fig)


def fig_feature_distributions(df: pd.DataFrame, features: list[str], label_col: Optional[str] = "label",
                              log_features: Optional[list[str]] = None, ncols: int = 3):
    log_features = log_features or []
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, f in zip(axes.ravel(), features):
        values = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        use_log = f in log_features
        if use_log:
            values = np.log10(values.where(values > 0))
        groups = [(None, values)] if label_col is None or label_col not in df else [
            (g, values[df[label_col] == g]) for g in df[label_col].dropna().unique()]
        lo, hi = np.nanpercentile(values.dropna(), [0.5, 99.5]) if values.notna().any() else (0, 1)
        bins = np.linspace(lo, hi, 36)
        for g, v in groups:
            ax.hist(v.dropna().clip(lo, hi), bins=bins, histtype="step", linewidth=1.8,
                    color=CLASS_COLORS.get(g, PRIMARY), label=g, density=True)
        ax.set_title(("log10 " if use_log else "") + f, fontsize=10)
        ax.set_yticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=True)
    fig.suptitle("Feature distributions by disposition (density, 0.5-99.5 percentile range)", y=1.0, color=TEXT)
    return _finish(fig)


def fig_correlation_heatmap(df: pd.DataFrame, features: list[str], log_features: Optional[list[str]] = None):
    data = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for f in (log_features or []):
        if f in data:
            data[f] = np.log10(data[f].where(data[f] > 0))
    corr = data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1.0 * len(features) + 3, 0.85 * len(features) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.diverging_palette(35, 230, as_cmap=True), center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor=BG, annot_kws={"color": "#0b0b0b", "fontsize": 9},
                cbar_kws={"label": "Spearman correlation"})
    ax.set_title("Feature correlation (Spearman, pairwise complete)")
    ax.grid(False)
    return _finish(fig)


# --------------------------------------------------------------------------------------
# ANN training
# --------------------------------------------------------------------------------------
def fig_learning_curves(history: dict, best_epoch: Optional[int] = None, title: str = "ANN learning curves"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, (k_tr, k_va, name) in zip(axes, [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]):
        if k_tr not in history:
            ax.axis("off")
            continue
        ax.plot(epochs, history[k_tr], color=STRATEGY_COLORS["A"], label=f"Training {name.lower()}")
        ax.plot(epochs, history[k_va], color=STRATEGY_COLORS["B"], linestyle="--", label=f"Validation {name.lower()}")
        if best_epoch:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")
            ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", color=MUTED, va="top", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name} per epoch")
        ax.legend(loc="best")
    fig.suptitle(title, color=TEXT, fontweight="bold")
    return _finish(fig)


def fig_confusion_matrix(cm: pd.DataFrame, title: str = "Confusion matrix", normalise: bool = True):
    values = cm.to_numpy(dtype=float)
    shown = values / values.sum(axis=1, keepdims=True).clip(min=1) if normalise else values
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(shown, cmap=sns.light_palette(PRIMARY, as_cmap=True), vmin=0, vmax=max(shown.max(), 1e-9))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = f"{int(values[i, j]):,}" + (f"\n{shown[i, j]:.0%}" if normalise else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color=BG if shown[i, j] > 0.45 else TEXT)
    ax.set_xticks(range(values.shape[1]), [c.replace("Pred ", "") for c in cm.columns], rotation=20, ha="right")
    ax.set_yticks(range(values.shape[0]), [c.replace("True ", "") for c in cm.index])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (catalogue disposition)")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    return _finish(fig)


# --------------------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------------------
def fig_metric_comparison(table: pd.DataFrame, metric: str, title: Optional[str] = None):
    """Grouped bars: models on x, colour = strategy. Values printed on the bars."""
    models = list(dict.fromkeys(table["model"]))
    strategies = [s for s in "ABC" if s in set(table["strategy_key"])]
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(models) + 2), 4))
    for i, s in enumerate(strategies):
        sub = table[table.strategy_key == s].set_index("model")[metric].reindex(models)
        x = np.arange(len(models)) + (i - (len(strategies) - 1) / 2) * width
        bars = ax.bar(x, sub.values, width=width * 0.92, color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
        for b, v in zip(bars, sub.values):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    top = np.nanmax(table[metric].to_numpy(float))
    ax.set_ylim(0, top * 1.15 if np.isfinite(top) and top > 0 else 1)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} by model and strategy (common test set)")
    ax.legend(title="Training strategy", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(strategies),
              frameon=False)
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def _curve_panels(results: dict, kind: str, positive_index: int, y_test):
    from .evaluation import pr_curve_data, reliability_curve_data, roc_curve_data
    keys = [k for k in "ABC" if k in results]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for m in results[k].models:
            color, ls, _ = _model_style(m.model_name)
            if kind == "roc":
                c = roc_curve_data(y_test, m.proba_test, positive_index)
                if c is not None:
                    ax.plot(c[0], c[1], color=color, linestyle=ls, label=f"{m.model_name} (AUC {m.metrics['roc_auc']:.3f})")
            elif kind == "pr":
                c = pr_curve_data(y_test, m.proba_test, positive_index)
                if c is not None:
                    ax.plot(c[0], c[1], color=color, linestyle=ls, label=f"{m.model_name} (AP {m.metrics['pr_auc']:.3f})")
            else:
                c = reliability_curve_data(y_test, m.proba_test, positive_index)
                if c is not None:
                    ax.plot(c[0], c[1], color=color, linestyle=ls, marker=_model_style(m.model_name)[2], markersize=4,
                            label=f"{m.model_name} (ECE {m.metrics['ece']:.3f})")
        ax.set_title(STRATEGY_LABELS[k], fontsize=10)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
            ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate")
        elif kind == "pr":
            ax.set_xlabel("Recall"), ax.set_ylabel("Precision")
        else:
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
            ax.set_xlabel("Mean predicted P(planet-like)"), ax.set_ylabel("Observed fraction planet-like")
        ax.legend(fontsize=7.5, loc="lower right" if kind != "pr" else "lower left")
    return fig


def fig_roc(results, y_test, positive_index=1):
    fig = _curve_panels(results, "roc", positive_index, y_test)
    fig.suptitle("ROC curves on the common test set", color=TEXT, fontweight="bold")
    return _finish(fig)


def fig_pr(results, y_test, positive_index=1):
    fig = _curve_panels(results, "pr", positive_index, y_test)
    fig.suptitle("Precision-Recall curves on the common test set", color=TEXT, fontweight="bold")
    return _finish(fig)


def fig_reliability(results, y_test, positive_index=1):
    fig = _curve_panels(results, "rel", positive_index, y_test)
    fig.suptitle("Reliability (calibration) diagrams - closer to the dotted diagonal is better calibrated",
                 color=TEXT, fontweight="bold")
    return _finish(fig)


# --------------------------------------------------------------------------------------
# Population analysis
# --------------------------------------------------------------------------------------
def fig_distribution_overlay(values, weight_vectors: dict, xlabel: str, log_x: bool = True, bins: int = 50):
    values = np.asarray(values, float)
    ok = np.isfinite(values) & (values > 0 if log_x else True)
    lo, hi = np.percentile(values[ok], [0.5, 99.5])
    edges = np.logspace(np.log10(lo), np.log10(hi), bins) if log_x else np.linspace(lo, hi, bins)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, w in weight_vectors.items():
        m = ok & (w > 0)
        ax.hist(values[m], bins=edges, weights=w[m] / w[m].sum(), histtype="step", linewidth=2,
                color=POP_COLORS.get(name, PRIMARY), label=f"{name} (n={int(m.sum()):,})")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of (weighted) records per bin")
    ax.set_title(f"{xlabel}: distribution under each label strategy")
    ax.legend()
    return _finish(fig)


def fig_box_by_strategy(values, weight_vectors: dict, ylabel: str, log_y: bool = True):
    """Box plot from weighted quantiles (whiskers = 5th-95th weighted percentile)."""
    from .population_analysis import _valid, weighted_quantile
    stats_ = []
    for name, w in weight_vectors.items():
        v, ww = _valid(values, w)
        q = weighted_quantile(v, ww, [0.05, 0.25, 0.5, 0.75, 0.95])
        stats_.append({"label": name.replace(" records", ""), "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3],
                       "whishi": q[4], "fliers": []})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bp = ax.bxp(stats_, showfliers=False, patch_artist=True, widths=0.5)
    for patch, name in zip(bp["boxes"], weight_vectors):
        patch.set_facecolor(POP_COLORS.get(name, PRIMARY))
        patch.set_alpha(0.55)
        patch.set_edgecolor(TEXT)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set_color(TEXT)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: weighted box plot (whiskers 5th-95th percentile)")
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def fig_ci_forest(summary: pd.DataFrame, stat: str = "median", xlabel: str = "Planet radius (R_earth)"):
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(summary) + 1.3))
    for i, row in enumerate(summary.itertuples(index=False)):
        r = row._asdict()
        color = POP_COLORS.get(r["strategy"], PRIMARY)
        ax.errorbar(r[stat], i, xerr=[[r[stat] - r[f"{stat}_ci_low"]], [r[f"{stat}_ci_high"] - r[stat]]],
                    fmt="o", color=color, capsize=4, markersize=7, markeredgecolor=BG, markeredgewidth=1.5)
        ax.text(r[stat], i - 0.14, f"{r[stat]:.3f}  [{r[f'{stat}_ci_low']:.3f}, {r[f'{stat}_ci_high']:.3f}]",
                color=TEXT, fontsize=8.5, va="bottom", ha="center")
    ax.set_yticks(range(len(summary)), list(summary["strategy"]))
    ax.set_ylim(len(summary) - 0.4, -0.7)
    ax.set_xlabel(f"{stat.title()} {xlabel} with 95% bootstrap CI")
    ax.set_title(f"{stat.title()} with bootstrap confidence intervals")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
    return _finish(fig)


def fig_period_radius_scatter(clean: pd.DataFrame, weight: Optional[np.ndarray] = None, title: str = ""):
    d = clean[["koi_period", "koi_prad", "label"]].copy()
    if weight is not None:
        d = d[np.asarray(weight) > 0]
    d = d[(d.koi_period > 0) & (d.koi_prad > 0)].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for g, sub in d.groupby("label"):
        ax.scatter(sub.koi_period, sub.koi_prad, s=7, alpha=0.45, color=CLASS_COLORS.get(g, PRIMARY),
                   label=f"{g} (n={len(sub):,})", edgecolors="none")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Planet radius (R_earth)")
    ax.set_title(title or "Orbital period vs planet radius (observed, detection-selected sample)")
    ax.legend(markerscale=2.5)
    return _finish(fig)


def fig_binned_trend(trends: dict[str, pd.DataFrame], xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, t in trends.items():
        if len(t):
            ax.plot(t.bin_mid, t.median_y, marker="o", color=POP_COLORS.get(name, PRIMARY), label=name)
    ax.set_xlabel(xlabel), ax.set_ylabel(f"Median {ylabel}")
    ax.set_yscale("log")
    ax.set_title(f"Median {ylabel} across {xlabel} bins")
    ax.legend()
    return _finish(fig)


def fig_correlation_forest(table: pd.DataFrame):
    labels = [f"{r.variable_x} vs {r.variable_y}" for r in table.drop_duplicates(["variable_x", "variable_y"]).itertuples()]
    strategies = list(dict.fromkeys(table.strategy))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(labels) * len(strategies) + 1.5))
    step = 0.8 / len(strategies)
    for si, s in enumerate(strategies):
        sub = table[table.strategy == s].reset_index(drop=True)
        y = np.arange(len(sub)) + (si - (len(strategies) - 1) / 2) * step
        ax.errorbar(sub.spearman_weighted, y, xerr=[sub.spearman_weighted - sub.ci_low, sub.ci_high - sub.spearman_weighted],
                    fmt="o", color=POP_COLORS.get(s, PRIMARY), capsize=3, markersize=5, label=s)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted Spearman correlation (95% bootstrap CI)")
    ax.set_title("Host-star property vs planet property correlations")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", visible=False)
    return _finish(fig)


def fig_prediction_summary(pred_table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    models = list(pred_table["model"])
    y = np.arange(len(models))
    axes[0].barh(y, pred_table["predicted_planet_like_share"], color=PRIMARY, height=0.55)
    for i, v in enumerate(pred_table["predicted_planet_like_share"]):
        axes[0].text(v, i, f" {v:.1%}", va="center", fontsize=8.5, color=TEXT)
    axes[0].set_yticks(y, models, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.15)
    axes[0].set_title("Share of test rows predicted planet-like")
    axes[0].grid(axis="y", visible=False)
    for col, lo, hi, off, label in (("median_hard", "median_hard_ci_low", "median_hard_ci_high", -0.12, "hard (argmax) selection"),
                                    ("median_prob_weighted", "median_prob_weighted_ci_low", "median_prob_weighted_ci_high", 0.12, "probability-weighted")):
        d = pred_table[[col, lo, hi]].astype(float)
        ok = d[col].notna()
        axes[1].errorbar(d.loc[ok, col], y[ok] + off, xerr=[d.loc[ok, col] - d.loc[ok, lo], d.loc[ok, hi] - d.loc[ok, col]],
                         fmt="o" if "hard" in col else "s", capsize=3, markersize=5,
                         color="#3987e5" if "hard" in col else "#d95926", label=label)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median planet radius (R_earth), 95% bootstrap CI")
    axes[1].set_title("Median radius of the predicted planet-like set")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", visible=False)
    return _finish(fig)


# --------------------------------------------------------------------------------------
# Uncertainty / robustness
# --------------------------------------------------------------------------------------
def fig_seed_stability(summary: pd.DataFrame, metric: str = "f1_macro"):
    """Mean +/- std across seeds for each model and strategy."""
    models = list(dict.fromkeys(summary["model"]))
    strategies = [s for s in summary["strategy"].unique()]
    key = {v: k for k, v in __import__("src.experiments", fromlist=["STRATEGY_NAMES"]).STRATEGY_NAMES.items()}
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(models) + 2), 4))
    width = 0.8 / len(strategies)
    for i, s in enumerate(strategies):
        sub = summary[summary.strategy == s].set_index("model").reindex(models)
        x = np.arange(len(models)) + (i - (len(strategies) - 1) / 2) * width
        ax.errorbar(x, sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"], fmt="o", capsize=4, markersize=7,
                    color=STRATEGY_COLORS.get(key.get(s, "A")), label=STRATEGY_LABELS.get(key.get(s, "A"), s),
                    markeredgecolor=BG, markeredgewidth=1.5)
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (mean +/- 1 SD over seeds)")
    ax.set_title("Seed-to-seed stability")
    ax.legend(loc="best")
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def fig_stability_forest(assess: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(assess) + 1.5))
    verdict_color = {"Stable improvement": "#199e70", "Stable decline": "#d95926"}
    for i, r in enumerate(assess.itertuples()):
        c = verdict_color.get(r.verdict, MUTED)
        if np.isfinite(r.ci_low):
            ax.errorbar(r.mean_diff, i, xerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", color=c,
                        capsize=4, markersize=7, markeredgecolor=BG)
        else:
            ax.plot(r.mean_diff, i, "o", color=c, markersize=7)
        ax.text(1.02, i, r.verdict, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_yticks(range(len(assess)), list(assess.comparison), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Paired difference in {METRIC_LABELS.get(assess.metric.iloc[0], assess.metric.iloc[0])} (95% CI)")
    ax.set_title("Which differences survive resampling?")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(right=0.72)
    return fig


def fig_sweep_lines(table: pd.DataFrame, x: str, metric: str, hue: str = "model", title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, sub in table.groupby(hue, sort=False):
        color, ls, marker = _model_style(name)
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[metric], color=color, linestyle=ls, marker=marker, label=name)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} vs {x.replace('_', ' ')}")
    ax.legend(fontsize=8)
    return _finish(fig)


def fig_grouped_bars(table: pd.DataFrame, category: str, metric: str, hue: str, title: str = "", colors: Optional[dict] = None):
    cats = list(dict.fromkeys(table[category]))
    hues = list(dict.fromkeys(table[hue]))
    palette = colors or {h: c for h, c in zip(hues, ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"])}
    width = 0.8 / len(hues)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(cats) + 2), 4))
    for i, h in enumerate(hues):
        sub = table[table[hue] == h].set_index(category)[metric].reindex(cats)
        x = np.arange(len(cats)) + (i - (len(hues) - 1) / 2) * width
        bars = ax.bar(x, sub.values, width=width * 0.92, color=palette.get(h, PRIMARY), label=str(h))
        for b, v in zip(bars, sub.values):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_xticks(np.arange(len(cats)), cats, rotation=15, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or METRIC_LABELS.get(metric, metric))
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def fig_before_after(raw_values, transformed_values, scaled_values, name: str, log_applied: bool):
    """Raw vs (log-)transformed vs imputed+standardised training distribution of one feature."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, vals, title in zip(
        axes, (raw_values, transformed_values, scaled_values),
        ("Raw (before)", "log10 applied" if log_applied else "No transform", "Imputed + standardised (after)"),
    ):
        v = pd.Series(np.asarray(vals, float)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v.clip(lo, hi), bins=40, color=PRIMARY, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
    axes[0].set_xlabel(name)
    fig.suptitle(f"{name}: effect of preprocessing (training split, 0.5-99.5 percentile)", color=TEXT, fontweight="bold")
    return _finish(fig)


# ---------------------------------------------------------------------------------------
# Results dashboard figures (computed results only)
# ---------------------------------------------------------------------------------------
# Axes of the radar: every axis is oriented so that OUTWARD = BETTER (error metrics are shown as 1 - value).
RADAR_AXES = [("accuracy", "Accuracy", False), ("f1_macro", "Macro-F1", False), ("roc_auc", "ROC-AUC", False),
              ("pr_auc", "PR-AUC", False), ("ece", "1 − ECE", True), ("brier", "1 − Brier", True)]


def fig_results_radar(table: pd.DataFrame, model: str):
    """One radar per model: a polygon per training strategy (outward = better on every axis)."""
    sub = table[table.model == model]
    labels = [a[1] for a in RADAR_AXES]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    fig = plt.figure(figsize=(5.6, 5.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(BG)
    for s in [k for k in "ABC" if k in set(sub.strategy_key)]:
        row = sub[sub.strategy_key == s].iloc[0]
        vals = []
        for key, _, invert in RADAR_AXES:
            v = float(row.get(key, np.nan))
            vals.append(np.nan if not np.isfinite(v) else (1 - v if invert else v))
        v = np.nan_to_num(np.array(vals), nan=0.0).clip(0, 1)
        ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=STRATEGY_COLORS[s], linewidth=2.2, label=STRATEGY_LABELS[s])
        ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=STRATEGY_COLORS[s], alpha=0.14)
    ax.set_xticks(ang, labels, color=TEXT, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color=MUTED, fontsize=7)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.set_title(f"{model}: reliability profile (outward = better)", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    return _finish(fig)


def fig_results_heatmap(table: pd.DataFrame):
    """Models x metrics heat-map. Colour = column-wise rank-normalised 'goodness' (error metrics inverted); cells show raw values."""
    from matplotlib.colors import LinearSegmentedColormap
    cols = [("accuracy", False), ("f1_macro", False), ("roc_auc", False), ("pr_auc", False),
            ("log_loss", True), ("brier", True), ("ece", True)]
    cols = [(c, inv) for c, inv in cols if c in table and table[c].notna().any()]
    t = table.copy()
    t["row"] = t["model"] + "  [" + t["strategy_key"] + "]"
    t = t.sort_values(["strategy_key", "model"]).set_index("row")
    raw = t[[c for c, _ in cols]].astype(float)
    good = raw.copy()
    for c, inv in cols:
        lo, hi = raw[c].min(), raw[c].max()
        z = (raw[c] - lo) / (hi - lo) if hi > lo else raw[c] * 0 + 0.5
        good[c] = 1 - z if inv else z
    cmap = LinearSegmentedColormap.from_list("exo", ["#101830", "#2C3A66", "#3987e5", "#6FD6FF", "#4DF0B0"])
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 3.2, 0.42 * len(raw) + 1.8))
    ax.imshow(good.to_numpy(float), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [METRIC_LABELS.get(c, c).split(" (")[0] for c, _ in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(raw)), raw.index)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="#050816" if good.iat[i, j] > 0.55 else TEXT)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Every model × every metric (brighter = better within that metric)")
    return _finish(fig)


def fig_size_vs_f1(table: pd.DataFrame):
    """Training-set size against macro-F1: shows what Experiment B's filtering costs in data."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, r in table.iterrows():
        color, _, marker = _model_style(r["model"])
        ax.scatter(r["n_train"], r["f1_macro"], s=95, marker=marker, color=STRATEGY_COLORS[r["strategy_key"]],
                   edgecolor=TEXT, linewidth=0.7, zorder=3)
    for m in dict.fromkeys(table["model"]):
        _, _, marker = _model_style(m)
        ax.scatter([], [], marker=marker, color=MUTED, edgecolor=TEXT, s=70, label=m)
    for s in [k for k in "ABC" if k in set(table.strategy_key)]:
        ax.scatter([], [], marker="o", color=STRATEGY_COLORS[s], s=70, label=STRATEGY_LABELS[s])
    ax.set_xlabel("Training rows used")
    ax.set_ylabel("Macro-F1 (common test set)")
    ax.set_title("Data used vs. performance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8)
    return _finish(fig)
