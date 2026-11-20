"""Page: Data Explorer - upload, validate and explore the KOI table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.app_helpers import page_header, show_df

from src import visualizations as viz
from src.app_helpers import (download_table, fmt_table, init_state, note, require_raw, setup_page, show_fig,
                             show_validation)
from src.data_loader import load_koi_csv
from src.data_validation import (DEFAULT_LOG_FEATURES, GROUP_COLUMN, NAME_COLUMN, class_counts, normalise_labels,
                                 validate_koi_dataframe)

setup_page("Data Explorer")
init_state()
page_header("Data Explorer", "Upload, validate and explore the NASA Kepler KOI table before any modelling.", 1, "Data")
DOWNSTREAM = ("data", "data_alt", "prep_cfg", "results", "population", "uncertainty", "saved_models", "strategy_params", "ann_cfg")


def _store(df: pd.DataFrame, name: str, signature: str) -> None:
    st.session_state["raw_df"] = df
    st.session_state["source_name"] = name
    st.session_state["source_sig"] = signature
    st.session_state["validation"] = validate_koi_dataframe(df)
    for key in DOWNSTREAM:
        st.session_state.pop(key, None)
    st.session_state["figures"] = {}


uploaded = st.file_uploader("Upload KOI CSV", type=["csv", "txt"])

try:
    if uploaded is not None:
        sig = f"upload:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("source_sig") != sig:
            _store(load_koi_csv(uploaded), uploaded.name, sig)
            st.success(f"Loaded **{uploaded.name}**.")
except ValueError as exc:
    st.error(f"Could not load the file: {exc}")
    st.stop()

df = require_raw()
report = st.session_state["validation"]

st.markdown(f"**Current file:** `{st.session_state.get('source_name')}`")
show_validation(report)
if not report.is_trainable:
    st.error("This file cannot be used for training yet (see the errors above). You can still explore it below.")

feats = report.available_features
explore = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan) if feats else pd.DataFrame(index=df.index)
if "koi_disposition" in df.columns:
    explore["label"] = normalise_labels(df["koi_disposition"])

tabs = st.tabs(["Overview", "Class distribution", "Missing & duplicates", "Statistics", "Feature distributions", "Correlation"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Default features found", f"{len(feats)} / 7")
    st.markdown("**First five records**")
    show_df(df.head(5))
    st.markdown("**Column names, types and completeness**")
    info = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
                         "non_null": df.notna().sum().values,
                         "missing_%": (df.isna().mean().values * 100).round(2)})
    show_df(info)

with tabs[1]:
    if "koi_disposition" not in df.columns:
        st.error("Column koi_disposition is missing.")
    else:
        counts = class_counts(df)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Classes present and record counts (before training)**")
            tbl = counts.rename("records").to_frame()
            tbl["share_%"] = (tbl["records"] / tbl["records"].sum() * 100).round(2)
            show_df(tbl)
            if "CONFIRMED" not in counts.index:
                note("No CONFIRMED rows are present. This is normal for the DR25 KOI table; the binary (CANDIDATE vs FALSE POSITIVE) "
                     "experiments are used and the three-class experiment is disabled.")
            supported = counts.drop("UNSUPPORTED / MISSING", errors="ignore")
            if len(supported) >= 2:
                ratio = supported.max() / supported.min()
                st.write(f"Imbalance ratio (largest / smallest class): **{ratio:.2f}**")
        with c2:
            show_fig(viz.fig_class_distribution(counts), "class_distribution", "explorer_cls")

with tabs[2]:
    n = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fully duplicated rows", f"{int(df.duplicated().sum()):,}")
    if NAME_COLUMN in df.columns:
        c2.metric("Duplicate KOI names", f"{int(df[NAME_COLUMN].duplicated().sum()):,}")
    if GROUP_COLUMN in df.columns:
        per_star = df[GROUP_COLUMN].value_counts()
        c3.metric("Host stars with >1 KOI", f"{int((per_star > 1).sum()):,}")
        st.caption("Stars hosting several KOIs are why the default split is grouped by kepid: objects around one star must not be split across train and test.")
    if feats:
        miss = explore[feats].isna().sum()
        show_fig(viz.fig_missing_values(miss, n), "missing_values", "explorer_missing")
    all_missing = df.isna().sum().rename("missing").to_frame()
    all_missing["missing_%"] = (all_missing["missing"] / n * 100).round(2)
    st.markdown("**All columns with missing values**")
    show_df(all_missing[all_missing["missing"] > 0].sort_values("missing", ascending=False))

with tabs[3]:
    if feats:
        desc = explore[feats].describe().T
        show_df(fmt_table(desc))
        download_table(desc.reset_index(names="feature"), "feature_statistics.csv", key="dl_stats")
    else:
        st.info("No numeric default features available.")

with tabs[4]:
    if feats:
        logs = [f for f in DEFAULT_LOG_FEATURES if f in feats]
        show_fig(viz.fig_feature_distributions(explore, feats, "label" if "label" in explore else None, logs),
                 "feature_distributions", "explorer_dist")
        st.caption("Period, radius and insolation are shown on a log10 scale because they span several orders of magnitude.")

with tabs[5]:
    if len(feats) >= 2:
        logs = [f for f in DEFAULT_LOG_FEATURES if f in feats]
        show_fig(viz.fig_correlation_heatmap(explore, feats, logs), "correlation_heatmap", "explorer_corr")
