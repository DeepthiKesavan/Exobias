"""Page: Model Comparison - ANN experiments vs traditional models on the common test set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_helpers import page_header, show_df

from src import visualizations as viz
from src.app_helpers import (download_table, fmt_table, init_state, note, require_data, require_results, setup_page,
                             show_fig, show_model_detail)
from src.evaluation import confusion_matrix_df
from src.experiments import STRATEGY_NAMES, results_to_frame

setup_page("Model Comparison")
init_state()
page_header("Model Comparison", "Compare the ANN experiments with traditional models on accuracy, macro-F1 and reliability metrics.", 4, "Evaluation")
data = require_data()
results = require_results()

full = results_to_frame(results, "full")
hc = results_to_frame(results, "hc")

note("All models are scored on the **same original held-out test set** (grouped by host star unless a fallback was reported). "
     "Experiment B's model is trained on a **smaller** population; the table shows each model's training size.", "note")
thresholds = {round(e.params.eval_threshold, 3) for e in results.values()}
if len(thresholds) > 1:
    st.warning(f"The experiments were trained with different confidence thresholds {sorted(thresholds)}; their "
               "'high-confidence test subset' metrics are not computed on the same rows. Retrain with one threshold for a fair comparison.")

metric_cols = ["model", "strategy", "n_train", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
               "log_loss", "brier", "ece", "roc_auc", "pr_auc"]
tabs = st.tabs(["Tables", "Accuracy & Macro-F1", "Reliability metrics", "Confusion matrices", "ROC / PR", "Calibration"])

with tabs[0]:
    st.markdown("**Full common test set**")
    table = full[[c for c in metric_cols if c in full]].sort_values("f1_macro", ascending=False)
    show_df(fmt_table(table))
    download_table(full, "model_comparison_full_test.csv", key="dl_cmp_full")
    if len(hc):
        st.markdown("**High-confidence part of the test set only** (rows whose koi_score satisfies the threshold rule)")
        show_df(fmt_table(hc[[c for c in metric_cols + ["n_samples"] if c in hc]].sort_values("f1_macro", ascending=False)))
        download_table(hc, "model_comparison_high_confidence_test.csv", key="dl_cmp_hc")
    st.caption("Rankings here come from a single seed/split. Differences of a few thousandths are within run-to-run noise – "
               "use the Uncertainty Analysis page to test whether a difference is stable.")

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        show_fig(viz.fig_metric_comparison(full, "accuracy"), "cmp_accuracy", "cmp_acc")
    with c2:
        show_fig(viz.fig_metric_comparison(full, "f1_macro"), "cmp_macro_f1", "cmp_f1")
    note("Accuracy alone can hide a weak minority-class performance; compare Macro-F1 and the reliability metrics before drawing conclusions.")

with tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        show_fig(viz.fig_metric_comparison(full, "log_loss"), "cmp_log_loss", "cmp_ll")
    with c2:
        show_fig(viz.fig_metric_comparison(full, "brier"), "cmp_brier", "cmp_br")
    with c3:
        show_fig(viz.fig_metric_comparison(full, "ece"), "cmp_ece", "cmp_ece")

with tabs[3]:
    labels = [f"{m.model_name} – {STRATEGY_NAMES[s]}" for s, e in results.items() for m in e.models]
    lookup = {f"{m.model_name} – {STRATEGY_NAMES[s]}": m for s, e in results.items() for m in e.models}
    pick = st.selectbox("Model", labels)
    show_model_detail(lookup[pick], data, "cmp_detail")

with tabs[4]:
    if data.n_classes != 2:
        st.info("ROC and Precision-Recall curves here are drawn for the binary task. For three classes see the macro one-vs-rest ROC-AUC in the tables.")
    else:
        show_fig(viz.fig_roc(results, data.y_test, data.positive_index), "cmp_roc", "cmp_roc")
        show_fig(viz.fig_pr(results, data.y_test, data.positive_index), "cmp_pr", "cmp_pr")
        st.caption(f"Positive class = {data.class_names[data.positive_index]}. The PR baseline equals the positive-class share of the test set.")

with tabs[5]:
    if data.n_classes == 2:
        show_fig(viz.fig_reliability(results, data.y_test, data.positive_index), "cmp_reliability", "cmp_rel")
    cal = full[["model", "strategy", "ece", "brier", "log_loss"]].sort_values("ece")
    show_df(fmt_table(cal))
    st.caption("ECE: top-label confidence vs accuracy in 10 equal-width bins (lower is better calibrated).")

st.markdown("### Does the ANN help?")
anns = full[full.model == "ANN"]
trad = full[full.model != "ANN"]
if len(anns) and len(trad):
    for key in sorted(set(anns.strategy_key)):
        a = anns[anns.strategy_key == key].iloc[0]
        best = trad[trad.strategy_key == key].sort_values("f1_macro", ascending=False).iloc[0]
        diff = a.f1_macro - best.f1_macro
        st.write(f"**{STRATEGY_NAMES[key]}** – ANN macro-F1 {a.f1_macro:.4f} vs best traditional model ({best.model}) {best.f1_macro:.4f} "
                 f"→ difference {diff:+.4f} (single run; no assumption that the ANN is better).")
else:
    st.info("Train the ANN and at least one traditional model to see this comparison.")
