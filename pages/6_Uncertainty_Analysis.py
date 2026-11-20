"""Page: Uncertainty Analysis - are the findings stable under different assumptions?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.app_helpers import page_header, show_df

from src import visualizations as viz
from src.ann_model import ANNConfig, tensorflow_available
from src.app_helpers import (SCORE_WARNING, download_table, fmt_table, init_state, note, require_data, setup_page, show_fig)
from src.baseline_models import available_baselines
from src.data_loader import load_koi_csv
from src.data_validation import normalise_labels
from src.experiments import STRATEGY_NAMES, StrategyParams
from src import uncertainty_analysis as ua

setup_page("Uncertainty Analysis")
init_state()
page_header("Uncertainty & Robustness Analysis", "Test whether findings survive different seeds, thresholds, weights, splits and feature sets.", 6, "Robustness")
data = require_data()
raw = st.session_state["raw_df"]
prep = st.session_state["prep_cfg"]
features, split_mode, base_seed = prep["features"], prep["split_mode"], prep["seed"]
params: StrategyParams = st.session_state.get("strategy_params") or StrategyParams()
unc = st.session_state["uncertainty"]
tf_ok = tensorflow_available()

note("Every number here comes from models actually trained on your data. A high accuracy alone is never treated as proof of reliability; "
     "a finding is called **stable** only when it survives resampling.", "note")
note(SCORE_WARNING, "warn")

# frame restricted to the classes chosen in preprocessing, so robustness runs use the same task
scoped = raw[normalise_labels(raw["koi_disposition"]).isin(prep["classes"])]

with st.expander("Run settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    n_seeds = int(c1.slider("Number of random seeds", 3, 10, 5))
    baselines = c2.multiselect("Traditional models", available_baselines(),
                               default=[m for m in ("Logistic Regression", "Random Forest") if m in available_baselines()])
    thresholds = c3.multiselect("Confidence thresholds", [0.60, 0.70, 0.80, 0.90, 0.95], default=[0.70, 0.80, 0.90])
    c4, c5 = st.columns(2)
    include_ann = c4.checkbox("Include the ANN (much slower)", value=False, disabled=not tf_ok)
    ann_epochs = int(c5.number_input("ANN max epochs in robustness runs", 5, 200, 30, disabled=not (tf_ok and include_ann)))
    if not tf_ok:
        st.caption("TensorFlow not installed: robustness runs use traditional models only.")
    if include_ann:
        st.warning("ANN runs repeat training for every seed × strategy; expect long run times.")

seeds = list(range(base_seed, base_seed + n_seeds))
ann_prev = st.session_state.get("ann_cfg_dict") or {}
ann_cfg = ANNConfig(**{**ann_prev, "hidden_units": tuple(ann_prev.get("hidden_units", (64, 32))), "max_epochs": ann_epochs}) if include_ann else None
common = dict(params=params, ann_config=ann_cfg, include_ann=include_ann and tf_ok, baselines=baselines)


def runner(label):
    bar = st.progress(0.0, text=f"{label}…")

    def cb(done, total, msg):
        bar.progress(min(done / total, 1.0), text=f"{label}: {msg} ({done}/{total})")
    return cb


tabs = st.tabs(["Seeds", "Thresholds", "Weights", "Grouped vs random", "Features", "Classes & populations", "Dataset versions", "Summary"])

with tabs[0]:
    st.markdown("Repeats the whole pipeline (new split **and** new initialisation) for each seed; reports mean ± SD and paired stability verdicts.")
    if st.button("Run multi-seed experiment", type="primary"):
        try:
            table = ua.multi_seed_experiment(scoped, features, seeds, split_mode, progress=runner("Seeds"), **common)
            unc["seed_table"] = table
            unc["seed_summary"] = ua.summarise_across_seeds(table)
        except Exception as exc:
            st.error(f"Multi-seed run failed: {exc}")
    if "seed_table" in unc:
        metric = st.selectbox("Metric", ["f1_macro", "accuracy", "log_loss", "brier", "ece", "roc_auc"], key="seed_metric")
        unc["stability"] = ua.stability_assessment(unc["seed_table"], metric)
        show_df(fmt_table(unc["seed_summary"], 4))
        download_table(unc["seed_summary"], "seed_summary.csv", key="dl_seed_sum")
        show_fig(viz.fig_seed_stability(unc["seed_summary"], metric), "seed_stability", "seed_stab")
        st.markdown("**Paired differences across seeds** (first minus second)")
        show_df(fmt_table(unc["stability"], 4))
        if len(unc["stability"]):
            show_fig(viz.fig_stability_forest(unc["stability"]), "stability_forest", "stab_forest")
        st.caption("With few seeds the interval is wide: 'No reliable difference' can mean low statistical power, not proof of equality.")

with tabs[1]:
    st.markdown("Experiment B at several thresholds. Every model is also scored on the **same** high-confidence test subset so results are comparable.")
    if not thresholds:
        st.info("Select at least one threshold in Run settings.")
    elif st.button("Run threshold sweep"):
        try:
            unc["threshold_sweep"] = ua.threshold_sweep(data, tuple(thresholds), progress=runner("Thresholds"), **common)
        except Exception as exc:
            st.error(f"Threshold sweep failed: {exc}")
    sweep = unc.get("threshold_sweep")
    if sweep is not None:
        show = sweep[["model", "test_subset", "threshold", "n_train", "train_fraction_of_baseline", "n_samples", "accuracy", "f1_macro", "log_loss", "ece"]]
        show_df(fmt_table(show, 4))
        download_table(sweep, "threshold_sweep.csv", key="dl_thr")
        full = sweep[(sweep.test_subset == "full test set") & sweep.threshold.notna()]
        if len(full):
            show_fig(viz.fig_sweep_lines(full, "threshold", "f1_macro", title="Macro-F1 vs threshold (full test set)"), "threshold_f1", "thr_f1")
            show_fig(viz.fig_sweep_lines(full, "train_fraction_of_baseline", "f1_macro", title="Macro-F1 vs fraction of training data kept"), "threshold_size", "thr_size")
        st.caption("Raising t removes more training rows: any change mixes label cleanliness with a smaller training set.")

with tabs[2]:
    st.markdown("Experiment C under different weight definitions (exponent, floor, missing-score policy) against the unweighted baseline.")
    if st.button("Run weight sensitivity"):
        try:
            unc["weight_sensitivity"] = ua.weight_sensitivity(data, progress=runner("Weights"), **common)
        except Exception as exc:
            st.error(f"Weight sensitivity failed: {exc}")
    ws = unc.get("weight_sensitivity")
    if ws is not None:
        show_df(fmt_table(ws[["model", "weight_setting", "n_train", "accuracy", "f1_macro", "log_loss", "brier", "ece"]], 4))
        download_table(ws, "weight_sensitivity.csv", key="dl_ws")
        show_fig(viz.fig_grouped_bars(ws, "weight_setting", "f1_macro", "model", "Macro-F1 by weight setting",
                                      colors={m: viz.MODEL_STYLE.get(m, ("#8da2b5",))[0] for m in ws["model"].unique()}),
                 "weight_sensitivity", "ws_bar")

with tabs[3]:
    st.markdown("Host-star-grouped split vs stratified row split, repeated over seeds. A stratified split can leak star-level information and look better.")
    if st.button("Run split comparison"):
        try:
            unc["split_comparison"] = ua.split_comparison(scoped, features, seeds[:max(3, min(n_seeds, 5))],
                                                          progress=runner("Splits"), **common)
        except Exception as exc:
            st.error(f"Split comparison failed: {exc}")
    sp = unc.get("split_comparison")
    if sp is not None:
        pivot = (sp.groupby(["model", "strategy", "split_requested"])[["accuracy", "f1_macro", "log_loss", "ece"]].mean().reset_index())
        show_df(fmt_table(pivot, 4))
        download_table(pivot, "split_comparison.csv", key="dl_split")
        wide = pivot.pivot_table(index=["model", "strategy"], columns="split_requested", values="f1_macro").reset_index()
        if {"grouped", "stratified"} <= set(wide.columns):
            wide["stratified_minus_grouped"] = wide["stratified"] - wide["grouped"]
            st.markdown("**Macro-F1: stratified minus grouped** (positive = random split looks more optimistic)")
            show_df(fmt_table(wide, 4))
        used = sp.groupby("split_requested")["split_used"].agg(lambda s: ", ".join(sorted(set(s))))
        st.caption(f"Split modes actually used: {used.to_dict()}")

with tabs[4]:
    st.markdown("Leave-one-feature-out plus planetary-only / stellar-only feature sets (baseline strategy A).")
    if st.button("Run feature sensitivity"):
        try:
            unc["feature_sensitivity"] = ua.feature_sensitivity(scoped, features, base_seed, split_mode, "A", progress=runner("Features"), **common)
        except Exception as exc:
            st.error(f"Feature sensitivity failed: {exc}")
    fs = unc.get("feature_sensitivity")
    if fs is not None:
        show_df(fmt_table(fs[["model", "feature_set", "n_features", "accuracy", "f1_macro", "f1_macro_change_vs_all", "log_loss"]], 4))
        download_table(fs, "feature_sensitivity.csv", key="dl_fs")
        show_fig(viz.fig_grouped_bars(fs, "feature_set", "f1_macro_change_vs_all", "model", "Change in macro-F1 vs all features",
                                      colors={m: viz.MODEL_STYLE.get(m, ("#8da2b5",))[0] for m in fs["model"].unique()}),
                 "feature_sensitivity", "fs_bar")

with tabs[5]:
    st.markdown("**Class distribution across splits and training cohorts**")
    cd = ua.class_distribution_comparison(data, params)
    show_df(fmt_table(cd, 4))
    unc["class_distribution"] = cd
    download_table(cd, "class_distribution_comparison.csv", key="dl_cd")
    st.markdown("**Observed median radius under different thresholds / weights** (population summary stability)")
    if "koi_prad" in features and st.button("Run population-summary stability"):
        try:
            unc["population_stability"] = ua.population_stability(scoped, features, tuple(thresholds or (0.8,)), "koi_prad")
        except Exception as exc:
            st.error(f"Population stability failed: {exc}")
    ps = unc.get("population_stability")
    if ps is not None:
        show_df(fmt_table(ps, 4))
        download_table(ps, "population_stability.csv", key="dl_ps")

with tabs[6]:
    st.markdown("Upload another KOI catalogue version/source to re-run the same pipeline on it (needs ≥ 2 datasets in total).")
    other = st.file_uploader("Second KOI CSV (optional)", type=["csv", "txt"], key="second_csv")
    if other is not None and st.button("Run dataset sensitivity"):
        try:
            second = load_koi_csv(other)
            unc["dataset_sensitivity"] = ua.dataset_sensitivity(
                {st.session_state.get("source_name", "dataset 1"): scoped, other.name: second}, features, base_seed, split_mode, **common)
        except Exception as exc:
            st.error(f"Dataset sensitivity failed: {exc}")
    dsx = unc.get("dataset_sensitivity")
    if dsx is not None:
        show_df(fmt_table(dsx[["dataset", "classes", "n_rows_clean", "model", "strategy", "n_train", "accuracy", "f1_macro", "log_loss", "ece"]], 4))
        download_table(dsx, "dataset_sensitivity.csv", key="dl_dsx")

with tabs[7]:
    st.markdown("### Which conclusions are stable?")
    stab = unc.get("stability")
    if stab is None or len(stab) == 0:
        st.info("Run the multi-seed experiment to get stable / unstable verdicts.")
    else:
        stable = stab[stab.verdict.isin(["Stable improvement", "Stable decline"])]
        unstable = stab[~stab.verdict.isin(["Stable improvement", "Stable decline"])]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Differences that survived resampling**")
            for r in stable.itertuples():
                st.write(f"- {r.comparison}: {r.verdict.lower()} (mean {r.mean_diff:+.4f}, {r.n_seeds} seeds)")
            if len(stable) == 0:
                st.write("None under the current settings.")
        with c2:
            st.markdown("**No reliable difference / insufficient evidence**")
            for r in unstable.itertuples():
                st.write(f"- {r.comparison}: {r.verdict.lower()} (mean {r.mean_diff:+.4f})")
        st.caption(f"Metric: {stab.metric.iloc[0]}. Verdicts come from paired per-seed differences with a 95% Student-t interval.")
