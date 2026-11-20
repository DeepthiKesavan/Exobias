from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .experiments import STRATEGY_NAMES, results_to_frame

LIMITATIONS = [
    "Catalogue dispositions are not perfect ground truth. A model trained and tested on them measures "
    "agreement with those dispositions, not necessarily the discovery of real planets.",
    "koi_score is a Robovetter vetting score, not a calibrated probability that an object is a real planet. "
    "The filtering (Experiment B) and weighting (Experiment C) are sensitivity-analysis heuristics.",
    "The KOI catalogue is an observed, detection-selected sample. Differences in its distributions are not "
    "differences in the intrinsic exoplanet population, and nothing here estimates occurrence rates (that would "
    "need detection-completeness and reliability corrections).",
    "ANN (and other model) predictions do not establish astronomical confirmation.",
    "Experiment B trains on a smaller population than the baseline; differences may reflect data quantity as well "
    "as label quality.",
    "KS p-values between nested strategy samples are approximate and not valid independent-sample tests; use the "
    "statistic D and paired bootstrap intervals descriptively.",
]


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}", max_rows: int = 60) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metric(results, strategy, model, metric, subset="metrics"):
    exp = results.get(strategy)
    m = exp.get(model) if exp else None
    return None if m is None else getattr(m, subset, None) and getattr(m, subset).get(metric)


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _question_answers(ctx: dict) -> list[str]:
    results = ctx.get("results") or {}
    stab = ctx.get("stability")
    out = []

    def stability_line(label_part: str) -> str:
        if stab is None or len(stab) == 0:
            return "Multi-seed stability has not been run, so this is a single-run observation only."
        rows = stab[stab.comparison.str.contains(label_part, regex=False)]
        if rows.empty:
            return "No matching multi-seed comparison available."
        return "Multi-seed verdicts: " + "; ".join(f"{r.comparison}: **{r.verdict}** (mean diff {r.mean_diff:+.4f}, {r.n_seeds} seeds)"
                                                  for r in rows.itertuples())

    a_ann, b_ann, c_ann = (_metric(results, s, "ANN", "f1_macro") for s in "ABC")
    out.append(f"**1. Does filtering lower-confidence records change ANN performance?** "
               + (f"Single run macro-F1: baseline {_fmt(a_ann)}, filtered {_fmt(b_ann)} (the filtered model used a smaller "
                  "training set). " if a_ann is not None and b_ann is not None else "ANN Experiments A/B not both available. ")
               + stability_line("ANN filtered"))
    out.append("**2. Does confidence-based sample weighting change the predictions?** "
               + (f"Single run macro-F1: baseline {_fmt(a_ann)}, weighted {_fmt(c_ann)}. " if a_ann is not None and c_ann is not None
                  else "ANN Experiments A/C not both available. ") + stability_line("ANN weighted"))
    ece = [_metric(results, s, "ANN", "ece") for s in "ABC"]
    out.append("**3. Does the strategy affect model stability?** "
               + ("Test-set ECE for the ANN: " + ", ".join(f"{s}={_fmt(e)}" for s, e in zip("ABC", ece)) + ". " if any(e is not None for e in ece) else "")
               + (f"Seed-to-seed standard deviations are in the robustness tables. " if ctx.get("seed_summary") is not None
                  else "Run the multi-seed experiment on the Uncertainty page to quantify stability."))
    pop = ctx.get("population", {})
    cmp_rad = pop.get("radius_comparison")
    out.append("**4. Does including/excluding uncertain records change planet-radius distributions?** "
               + ("; ".join(f"{r.comparison}: median difference {r.median_diff:+.3f} R_earth "
                            f"(95% CI {r.median_diff_ci_low:+.3f} to {r.median_diff_ci_high:+.3f}; "
                            f"{'excludes' if r.median_diff_ci_excludes_0 else 'includes'} 0), KS D={r.ks_D:.3f}, "
                            f"Cohen's d (log10)={r.cohens_d_log10:+.3f} [{r.effect_size_label}]" for r in cmp_rad.itertuples())
                  if cmp_rad is not None and len(cmp_rad) else "Population analysis not run yet."))
    cmp_per = pop.get("period_comparison")
    out.append("**5. Does it change orbital-period relationships?** "
               + ("; ".join(f"{r.comparison}: median period difference {r.median_diff:+.3f} d (95% CI {r.median_diff_ci_low:+.3f} to "
                            f"{r.median_diff_ci_high:+.3f}), KS D={r.ks_D:.3f}" for r in cmp_per.itertuples())
                  if cmp_per is not None and len(cmp_per) else "Population analysis not run yet."))
    split_tbl = ctx.get("split_comparison")
    out.append("**6. Are the patterns stable across validation methods?** "
               + ("A grouped-vs-stratified comparison was run; see the robustness table for per-strategy differences."
                  if split_tbl is not None and len(split_tbl) else "The grouped vs stratified comparison has not been run."))
    ann_rf = None
    if stab is not None and len(stab):
        r = stab[stab.comparison.str.startswith("ANN vs Random Forest")]
        ann_rf = None if r.empty else r.iloc[0]
    rf_f1 = _metric(results, "A", "Random Forest", "f1_macro")
    out.append("**7. Does the ANN offer any benefit over traditional models?** "
               + (f"Single run macro-F1 on the baseline cohort: ANN {_fmt(a_ann)}, Random Forest {_fmt(rf_f1)}, "
                  f"Logistic Regression {_fmt(_metric(results, 'A', 'Logistic Regression', 'f1_macro'))}. " if a_ann is not None else
                  "ANN results not available. ")
               + (f"Multi-seed verdict (ANN vs Random Forest): **{ann_rf.verdict}** (mean diff {ann_rf.mean_diff:+.4f}). " if ann_rf is not None else
                  "Multi-seed ANN comparison not run."))
    return out


def generate_report(ctx: dict) -> str:
    """``ctx`` keys (all optional): dataset_name, validation, cleaning_log, split_summary, split_mode, features,
    class_names, results, ann_summaries, population (dict of tables), stability, seed_summary,
    threshold_sweep, weight_sensitivity, split_comparison, feature_sensitivity."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = [
        "# ExoBias - Research Summary",
        f"_Generated {now}. Every number below comes from models and statistics computed in this session on the uploaded data._",
        "",
        "## 1. Research question",
        "How do different strategies for handling uncertain candidate labels (baseline, high-confidence filtering, "
        "confidence-weighting) affect machine-learning reliability and the interpretation of observed exoplanet population patterns?",
        "",
        "## 2. Data",
        f"* Source file: `{ctx.get('dataset_name', 'not recorded')}` (user-supplied NASA Exoplanet Archive KOI table)",
    ]
    if ctx.get("class_names"):
        L.append(f"* Classes used: {', '.join(ctx['class_names'])}")
    if ctx.get("features"):
        L.append(f"* Input features: {', '.join(ctx['features'])} (never koi_score, kepid, dispositions or vetting flags)")
    L.append(f"* Split mode used: {ctx.get('split_mode', 'n/a')}")
    for kind, msg in (ctx.get("validation") or []):
        L.append(f"* {kind.upper()}: {msg}")
    if ctx.get("cleaning_log") is not None:
        L += ["", "### Rows removed at each preprocessing stage", md_table(ctx["cleaning_log"], "{:.0f}")]
    if ctx.get("split_summary") is not None:
        L += ["", "### Data split", md_table(ctx["split_summary"], "{:.3f}")]

    results = ctx.get("results")
    if results:
        L += ["", "## 3. Experiments", ""]
        for s, exp in results.items():
            info = exp.cohort_info
            L.append(f"* **{STRATEGY_NAMES[s]}** - training rows {info['n_train']:,} of {info['n_train_before']:,} "
                     f"(removed {info['n_train_removed']:,}; by class {info['removed_by_class']}). "
                     + " ".join(info.get("notes", [])))
        full = results_to_frame(results, "full")
        cols = ["model", "strategy", "n_train", "accuracy", "balanced_accuracy", "f1_macro", "log_loss", "brier", "ece", "roc_auc", "pr_auc"]
        L += ["", "### Common held-out test set", md_table(full[[c for c in cols if c in full]])]
        hc = results_to_frame(results, "hc")
        if len(hc):
            L += ["", "### High-confidence part of the test set only", md_table(hc[[c for c in cols + ["n_samples"] if c in hc]])]
        for s, exp in results.items():
            ann = exp.get("ANN")
            if ann is not None and ann.ann_summary:
                a = ann.ann_summary
                L += ["", f"**ANN {STRATEGY_NAMES[s]}**: hidden layers {a['neurons_per_hidden_layer']}, best epoch {a['best_epoch']} of "
                          f"{a['epochs_run']}. " + " ".join(ann.observations)]
    else:
        L += ["", "## 3. Experiments", "_No experiments have been trained yet._"]

    pop = ctx.get("population") or {}
    L += ["", "## 4. Population analysis (observed, detection-selected catalogue)"]
    if pop:
        for name, tbl in pop.items():
            if tbl is not None and len(tbl):
                L += ["", f"### {name.replace('_', ' ').title()}", md_table(tbl)]
    else:
        L.append("_Not run yet._")

    L += ["", "## 5. Robustness"]
    any_rob = False
    for key, title in (("seed_summary", "Mean and SD over seeds"), ("stability", "Paired stability verdicts"),
                       ("threshold_sweep", "Confidence-threshold sweep"), ("weight_sensitivity", "Sample-weight sensitivity"),
                       ("split_comparison", "Grouped vs stratified split"), ("feature_sensitivity", "Feature sensitivity")):
        tbl = ctx.get(key)
        if tbl is not None and len(tbl):
            any_rob = True
            show = tbl[[c for c in tbl.columns if c not in ("caveat",)]]
            L += ["", f"### {title}", md_table(show, max_rows=40)]
    if not any_rob:
        L.append("_Not run yet._")

    L += ["", "## 6. Answers to the research questions (data-driven, hedged)"]
    L += [f"{a}\n" for a in _question_answers(ctx)]
    L += ["## 7. Limitations"] + [f"* {t}" for t in LIMITATIONS]
    return "\n".join(L) + "\n"


def results_bundle_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def save_tables(tables: dict[str, pd.DataFrame], directory: Path) -> list[str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, tbl in tables.items():
        if tbl is not None and len(tbl):
            path = directory / f"{name}.csv"
            tbl.to_csv(path, index=False)
            saved.append(str(path))
    return saved


def save_report(markdown: str, directory: Path, name: str = "exobias_research_summary") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
