"""Page: Results and Report - downloads and the generated research summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import visualizations as viz
from src.app_helpers import leaderboard, page_header, show_df, show_fig

from src.app_helpers import (MODELS_DIR, RESULTS_DIR, csv_bytes, download_table, fmt_table, init_state, note, require_data,
                             setup_page)
from src.experiments import STRATEGY_NAMES, results_to_frame
from src.report_generator import generate_report, results_bundle_zip, save_report, save_tables

setup_page("Results & Report")
init_state()
page_header("Results and Report", "Review overall results, download tables and graphs, and generate the research summary.", 7, "Output")
data = require_data()
report = st.session_state["validation"]
results = st.session_state.get("results", {})
population = st.session_state.get("population", {})
unc = st.session_state.get("uncertainty", {})
figures = st.session_state.get("figures", {})

# ---- collect tables -----------------------------------------------------------------------
tables: dict[str, pd.DataFrame] = {}
if results:
    tables["model_comparison_full_test"] = results_to_frame(results, "full")
    hc = results_to_frame(results, "hc")
    if len(hc):
        tables["model_comparison_high_confidence_test"] = hc
for key, value in population.items():
    if isinstance(value, pd.DataFrame) and len(value):
        tables[f"population_{key}"] = value
for key, value in unc.items():
    if isinstance(value, pd.DataFrame) and len(value):
        tables[f"uncertainty_{key}"] = value
tables["preprocessing_removal_log"] = data.cleaning.removal_log
tables["split_summary"] = data.split_summary()

# ---- overview -----------------------------------------------------------------------------
st.markdown("### Overall experiment results")
if results:
    full = tables["model_comparison_full_test"]
    show_df(fmt_table(full[["model", "strategy", "n_train", "accuracy", "f1_macro", "log_loss", "brier", "ece", "roc_auc"]]
                           .sort_values("f1_macro", ascending=False), 4))
else:
    st.info("No trained experiments yet (ANN Workbench). Preprocessing tables are still available below.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Experiments trained", len(results))
c2.metric("Population tables", sum(isinstance(v, pd.DataFrame) for v in population.values()))
c3.metric("Robustness tables", sum(isinstance(v, pd.DataFrame) for v in unc.values()))
c4.metric("Graphs captured", len(figures))

if results:
    st.markdown("### Results visualised")
    st.caption("Everything below is drawn from the experiments you trained in this session (single seed and split).")
    l1, l2 = st.columns(2)
    with l1:
        leaderboard(full, "f1_macro", "Macro-F1 (higher is better)")
    with l2:
        leaderboard(full, "accuracy", "Accuracy (higher is better)")
    v1, v2 = st.columns(2)
    with v1:
        radar_model = st.selectbox("Model for the reliability radar", sorted(set(full.model)), key="radar_model")
        show_fig(viz.fig_results_radar(full, radar_model), f"results_radar_{radar_model}", "res_radar")
    with v2:
        st.markdown("&nbsp;")
        show_fig(viz.fig_size_vs_f1(full), "results_size_vs_macro_f1", "res_size")
    show_fig(viz.fig_results_heatmap(full), "results_heatmap_all_models_metrics", "res_heat")

tabs = st.tabs(["CSV downloads", "Graphs (PNG)", "Research summary", "Saved models", "Write to results/ folder"])

with tabs[0]:
    st.caption("Model comparison, population analysis and robustness tables, as computed in this session.")
    for name, tbl in tables.items():
        c1, c2 = st.columns([4, 1])
        c1.write(f"`{name}.csv` – {len(tbl):,} rows")
        c2.download_button("Download CSV", csv_bytes(tbl), file_name=f"{name}.csv", mime="text/csv", key=f"dl_all_{name}")
    if tables:
        bundle = results_bundle_zip({f"{n}.csv": csv_bytes(t) for n, t in tables.items()})
        st.download_button("Download All tables (ZIP)", bundle, file_name="exobias_tables.zip", mime="application/zip")

with tabs[1]:
    if not figures:
        st.info("Graphs appear here after you view them on other pages (each chart is captured when it is drawn).")
    else:
        picked = st.multiselect("Select graphs", sorted(figures), default=[])
        for name in picked:
            st.image(figures[name], caption=name)
            st.download_button("Download PNG", figures[name], file_name=f"{name}.png", mime="image/png", key=f"dl_fig_{name}")
        if picked:
            st.download_button("Download Selected graphs (ZIP)", results_bundle_zip({f"{n}.png": figures[n] for n in picked}),
                               file_name="exobias_graphs.zip", mime="application/zip")

with tabs[2]:
    ctx = {
        "dataset_name": st.session_state.get("source_name"), "validation": report.as_messages(),
        "cleaning_log": data.cleaning.removal_log, "split_summary": data.split_summary(),
        "split_mode": data.split.mode_used, "features": data.features, "class_names": data.class_names,
        "results": results,
        "population": {k: v for k, v in population.items() if isinstance(v, pd.DataFrame)},
        "stability": unc.get("stability"), "seed_summary": unc.get("seed_summary"),
        "threshold_sweep": unc.get("threshold_sweep"), "weight_sensitivity": unc.get("weight_sensitivity"),
        "split_comparison": unc.get("split_comparison"), "feature_sensitivity": unc.get("feature_sensitivity"),
    }
    if st.button("Generate research summary", type="primary"):
        st.session_state["report_md"] = generate_report(ctx)
    md = st.session_state.get("report_md")
    if md:
        st.download_button("Download Research summary (Markdown)", md.encode("utf-8"), file_name="exobias_research_summary.md", mime="text/markdown")
        st.markdown(md)
    else:
        st.info("Click **Generate research summary**. The report only contains analyses you have actually run and says so for anything missing.")

with tabs[3]:
    saved = st.session_state.get("saved_models")
    if saved:
        st.success(f"Last save: `{saved['directory']}`")
    runs = sorted([p for p in MODELS_DIR.glob("run_*") if p.is_dir()], reverse=True) if MODELS_DIR.exists() else []
    if not runs:
        st.info("No saved models yet. Use **Save models + preprocessor** on the ANN Workbench page.")
    else:
        run = st.selectbox("Saved run", [r.name for r in runs])
        files = sorted(f for f in (MODELS_DIR / run).iterdir() if f.is_file())
        for f in files:
            c1, c2 = st.columns([4, 1])
            c1.write(f"`{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
            c2.download_button("Download", f.read_bytes(), file_name=f.name, key=f"dl_model_{run}_{f.name}")
        st.caption("Each run folder holds the models, preprocessor.joblib (fitted imputer + scaler), feature names and experiment_config.json.")

with tabs[4]:
    st.write(f"Writes CSVs to `{RESULTS_DIR / 'metrics'}`, graphs to `{RESULTS_DIR / 'graphs'}` and the summary to `{RESULTS_DIR / 'reports'}`.")
    if st.button("Write everything to the results/ folder"):
        try:
            n_csv = len(save_tables(tables, RESULTS_DIR / "metrics"))
            graphs_dir = RESULTS_DIR / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            for name, png in figures.items():
                (graphs_dir / f"{name}.png").write_bytes(png)
            path = save_report(st.session_state.get("report_md") or generate_report(ctx), RESULTS_DIR / "reports")
            st.success(f"Wrote {n_csv} CSV files, {len(figures)} graphs and `{path.name}`.")
        except Exception as exc:
            st.error(f"Could not write results: {exc}")
