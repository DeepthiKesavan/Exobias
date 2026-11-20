"""Page: Population Analysis - observed-catalogue patterns under each candidate-label strategy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.app_helpers import page_header, show_df

from src import visualizations as viz
from src.app_helpers import (SELECTION_WARNING, download_table, fmt_table, init_state, note, require_data, setup_page,
                             show_fig)
from src.experiments import STRATEGY_NAMES, StrategyParams
from src.population_analysis import (ALL_KEY, binned_median, compare_label_groups, compare_to_reference, correlation_table,
                                     prediction_based_summary, strategy_weight_vectors, summary_table)

setup_page("Population Analysis")
init_state()
page_header("Population Analysis", "Observed radius, period and host-star patterns under each candidate-label strategy.", 5, "Science")
data = require_data()
clean = data.clean
note(SELECTION_WARNING, "warn")
st.caption("Primary analysis uses observed catalogue values only. ANN-derived results are shown separately and are exploratory.")

prev = st.session_state.get("strategy_params") or StrategyParams()
c1, c2, c3 = st.columns(3)
threshold = c1.slider("High-confidence threshold t", 0.50, 0.95, float(prev.threshold), 0.05)
n_boot = int(c2.select_slider("Bootstrap resamples", [100, 200, 500, 1000], value=500))
c3.markdown("**Weights:** floor {:.2f}, exponent {:g}, missing score → {}".format(prev.min_weight, prev.weight_exponent, prev.missing_score_policy))

has_score = clean["score"].notna().any()
if not has_score:
    st.warning("koi_score is unavailable: only the 'All eligible records' strategy can be analysed.")

if st.button("Run population analysis", type="primary"):
    params = StrategyParams(threshold=float(threshold), min_weight=prev.min_weight, weight_exponent=prev.weight_exponent,
                            missing_score_policy=prev.missing_score_policy, neutral_weight=prev.neutral_weight)
    try:
        with st.spinner("Computing weighted statistics and bootstrap intervals…"):
            wv = strategy_weight_vectors(clean, params)
            out = {"weights": wv, "params": params, "n_boot": n_boot}
            for col, tag in (("koi_prad", "radius"), ("koi_period", "period")):
                if col in clean.columns:
                    vals = clean[col].to_numpy(float)
                    out[f"{tag}_summary"] = summary_table(vals, wv, n_boot=n_boot)
                    out[f"{tag}_comparison"] = compare_to_reference(vals, wv, ALL_KEY, n_boot=n_boot) if len(wv) > 1 else pd.DataFrame()
                    out[f"{tag}_groups"] = compare_label_groups(clean, col, wv, n_boot=n_boot)
            pairs = [(s, p) for s in ("koi_steff", "koi_srad", "koi_smass") for p in ("koi_prad", "koi_period")]
            out["host_correlations"] = correlation_table(clean, pairs, wv, n_boot=min(n_boot, 300))
            st.session_state["population"] = out
    except Exception as exc:
        st.error(f"Population analysis failed: {exc}")

pop = st.session_state.get("population")
if not pop:
    st.info("Click **Run population analysis** to compute the tables and charts.")
    st.stop()
wv = pop["weights"]

tabs = st.tabs(["Planet radius", "Orbital period", "Radius vs period", "Host star", "Candidate vs False Positive",
                "ANN predictions (exploratory)", "Statistical comparison"])


def distribution_tab(col, tag, label, log):
    if f"{tag}_summary" not in pop:
        st.info(f"Column {col} is not in the selected features/data.")
        return
    vals = clean[col].to_numpy(float)
    c1, c2 = st.columns(2)
    with c1:
        show_fig(viz.fig_distribution_overlay(vals, wv, label, log_x=log), f"{tag}_distribution", f"{tag}_dist")
    with c2:
        show_fig(viz.fig_box_by_strategy(vals, wv, label, log_y=log), f"{tag}_box", f"{tag}_box")
    st.markdown("**Summary statistics** (weighted; n_effective is the Kish effective sample size)")
    summary = pop[f"{tag}_summary"]
    show_df(fmt_table(summary, 3))
    download_table(summary, f"{tag}_summary.csv", key=f"dl_{tag}_sum")
    if "median_ci_low" in summary:
        show_fig(viz.fig_ci_forest(summary.dropna(subset=["median_ci_low"]), "median", label), f"{tag}_median_ci", f"{tag}_ci")


with tabs[0]:
    distribution_tab("koi_prad", "radius", "Planet radius (R_earth)", True)
with tabs[1]:
    distribution_tab("koi_period", "period", "Orbital period (days)", True)

with tabs[2]:
    if {"koi_period", "koi_prad"} <= set(clean.columns):
        names = list(wv)
        choice = st.selectbox("Strategy shown", names)
        show_fig(viz.fig_period_radius_scatter(clean, wv[choice], f"Period vs radius – {choice}"), "period_radius", "pr_scatter")
        try:
            import plotly.express as px
            d = clean[["koi_period", "koi_prad", "label"]].dropna()
            d = d[(d.koi_period > 0) & (d.koi_prad > 0) & (wv[choice][d.index] > 0)]
            fig = px.scatter(d, x="koi_period", y="koi_prad", color="label", log_x=True, log_y=True, opacity=0.5,
                             template="plotly_dark", color_discrete_map=viz.CLASS_COLORS)
            st.markdown("**Interactive version** (zoom / hover)")
            st.plotly_chart(fig)
        except ImportError:
            st.caption("Install plotly for the interactive scatter plot.")
        st.caption("Orbital period and radius are strongly shaped by transit-detection bias (short periods and large radii are easier to detect).")
    else:
        st.info("koi_period and koi_prad are both required.")

with tabs[3]:
    corr = pop["host_correlations"]
    if len(corr):
        show_fig(viz.fig_correlation_forest(corr), "host_star_correlations", "host_corr")
        show_df(fmt_table(corr, 3))
        download_table(corr, "host_star_correlations.csv", key="dl_host_corr")
        st.markdown("**Median planet radius across host-star bins**")
        var = st.selectbox("Host-star property", [c for c in ("koi_steff", "koi_srad", "koi_smass") if c in clean.columns])
        if "koi_prad" in clean.columns:
            x = clean[var].to_numpy(float)
            edges = np.unique(np.nanpercentile(x, np.linspace(0, 100, 9)))
            trends = {k: binned_median(x, clean["koi_prad"].to_numpy(float), w, edges) for k, w in wv.items()}
            show_fig(viz.fig_binned_trend(trends, var, "planet radius (R_earth)"), "host_binned_trend", "host_trend")
    else:
        st.info("No host-star columns (koi_steff, koi_srad, koi_smass) are present.")

with tabs[4]:
    for tag, label in (("radius", "Planet radius"), ("period", "Orbital period")):
        tbl = pop.get(f"{tag}_groups")
        if tbl is not None and len(tbl):
            st.markdown(f"**{label}: CANDIDATE vs FALSE POSITIVE within each strategy** (independent groups)")
            show_df(fmt_table(tbl, 4))
            download_table(tbl, f"{tag}_candidate_vs_false_positive.csv", key=f"dl_grp_{tag}")
    show_fig(viz.fig_period_radius_scatter(clean, None, "All eligible records by disposition"), "scatter_by_label", "grp_scatter")

with tabs[5]:
    note("Exploratory: model predictions measure agreement with catalogue dispositions and are **not** astronomical confirmation. "
         "Only held-out test rows are used so predictions are not in-sample.", "warn")
    results = st.session_state.get("results") or {}
    probas = {f"{m.model_name} [{STRATEGY_NAMES[s].split(':')[0]}]": m.proba_test
              for s, e in results.items() for m in e.models if m.model_name in ("ANN", "Random Forest")}
    if not probas:
        st.info("Train experiments in the ANN Workbench first.")
    elif "koi_prad" not in data.test.columns:
        st.info("koi_prad must be among the selected features.")
    else:
        pred = prediction_based_summary(data.test, probas, data.class_names, "koi_prad", n_boot=min(pop["n_boot"], 300))
        show_df(fmt_table(pred, 3))
        show_fig(viz.fig_prediction_summary(pred), "prediction_based_summary", "pred_sum")
        download_table(pred, "prediction_based_population_summary.csv", key="dl_pred")
        pop["prediction_summary"] = pred
        st.caption("Hard selection = rows whose argmax class is planet-like; probability-weighted = rows weighted by P(planet-like).")

with tabs[6]:
    for tag, label in (("radius", "Planet radius"), ("period", "Orbital period")):
        tbl = pop.get(f"{tag}_comparison")
        if tbl is not None and len(tbl):
            st.markdown(f"**{label}: each strategy vs all eligible records** (paired bootstrap, nested samples)")
            show_df(fmt_table(tbl.drop(columns=["caveat"]), 4))
            st.caption(tbl["caveat"].iloc[0])
            download_table(tbl, f"{tag}_strategy_comparison.csv", key=f"dl_cmp_{tag}")
    st.markdown(
        "* **Difference in medians** with a 95% paired bootstrap CI; if the CI includes 0 the data do not show a reliable shift.\n"
        "* **KS D** is the largest gap between the two weighted cumulative distributions (0 = identical).\n"
        "* **Cohen's d (log10)** is a standardised mean difference; size words are conventional, not astrophysical thresholds."
    )
