"""Page: Data Preprocessing - feature selection, cleaning, leakage-safe splitting and scaling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.app_helpers import page_header, show_df

from src import visualizations as viz
from src.app_helpers import (download_table, fmt_table, init_state, note, require_raw, setup_page, show_fig)
from src.data_validation import CLASS_ORDER, DEFAULT_LOG_FEATURES, candidate_feature_columns
from src.preprocessing import prepare_data

setup_page("Data Preprocessing")
init_state()
page_header("Data Preprocessing", "Select features, clean the table, split by host star, and fit the imputer and scaler on training data only.", 2, "Data")
df = require_raw()
report = st.session_state["validation"]
if not report.is_trainable:
    st.error("The loaded file failed validation (see Data Explorer). Fix the file before preprocessing.")
    st.stop()

cfg = st.session_state.get("prep_cfg", {})
allowed = candidate_feature_columns(df)
st.caption("The target, koi_score, kepid, names and vetting flags (koi_fpflag_*) are excluded from this list by design (anti-leakage).")

default_feats = [f for f in (cfg.get("features") or report.available_features) if f in allowed]
c1, c2 = st.columns(2)
features = c1.multiselect("Input features", allowed, default=default_feats)
log_default = [f for f in cfg.get("log_features", DEFAULT_LOG_FEATURES) if f in features]
log_features = c2.multiselect("Apply log10 before scaling (skewed features)", features, default=log_default)

c3, c4, c5 = st.columns(3)
modes = ["grouped", "stratified"]
split_mode = c3.radio("Primary split", modes, index=modes.index(cfg.get("split_mode", "grouped")),
                      format_func=lambda m: "Grouped by host star (kepid) – recommended" if m == "grouped" else "Stratified row split")
seed = int(c4.number_input("Random seed", min_value=0, max_value=10_000, value=int(cfg.get("seed", 42)), step=1))

supported = report.supported_classes
if report.supports_three_class:
    use_confirmed = c5.checkbox("Include CONFIRMED (three-class experiment)", value=cfg.get("three_class", False))
    classes = CLASS_ORDER if use_confirmed else ["FALSE POSITIVE", "CANDIDATE"]
else:
    classes = list(supported)
    c5.markdown("**Classes:** " + ", ".join(classes))
    c5.caption("Only genuinely present classes are used.")

if st.button("Run preprocessing", type="primary"):
    if len(features) < 2:
        st.error("Select at least two features.")
    else:
        try:
            with st.spinner("Cleaning, splitting and fitting the imputer/scaler on the training split…"):
                data = prepare_data(df, features, split_mode, seed, classes, log_features)
                alt_mode = "stratified" if split_mode == "grouped" else "grouped"
                data_alt = prepare_data(df, features, alt_mode, seed, classes, log_features)
            for key in ("results", "population", "uncertainty", "saved_models"):
                st.session_state[key] = {} if key != "saved_models" else None
            st.session_state["data"], st.session_state["data_alt"] = data, data_alt
            st.session_state["prep_cfg"] = {"features": features, "log_features": log_features, "split_mode": split_mode,
                                            "seed": seed, "three_class": len(classes) == 3, "classes": classes}
            st.success("Preprocessing complete. Earlier experiment results were cleared because the test set changed.")
        except Exception as exc:  # surfaced to the user rather than crashing the page
            st.error(f"Preprocessing failed: {exc}")

data = st.session_state.get("data")
if data is None:
    st.info("Choose settings and click **Run preprocessing**.")
    st.stop()
alt = st.session_state["data_alt"]

for w in data.warnings:
    st.warning(w)

tabs = st.tabs(["Rows removed", "Split & leakage check", "Class imbalance", "Before / after", "Fitted objects"])

with tabs[0]:
    st.markdown("**Rows removed at every preprocessing stage**")
    show_df(data.cleaning.removal_log)
    for n_ in data.cleaning.notes:
        st.info(n_)
    c1, c2, c3 = st.columns(3)
    c1.metric("Infinite values → NaN", data.cleaning.inf_replaced)
    c2.metric("Non-numeric → NaN", data.cleaning.coerced_to_nan)
    c3.metric("Rows kept for modelling", f"{len(data.clean):,}")
    st.caption("Missing numeric values are imputed with the **training-set median** after the split, so no validation/test information leaks.")


def shared_stars(d) -> tuple[int, int, int]:
    g = [set(x["group"].dropna()) for x in (d.train, d.val, d.test)]
    return len(g[0] & g[2]), len(g[0] & g[1]), len(g[1] & g[2])


with tabs[1]:
    for title, d in (("Primary split", data), ("Comparison split", alt)):
        st.markdown(f"**{title} – requested `{d.split.mode_requested}`, used `{d.split.mode_used}`**")
        show_df(fmt_table(d.split_summary(), 3))
        tr_te, tr_va, va_te = shared_stars(d)
        c1, c2, c3 = st.columns(3)
        c1.metric("Host stars shared train↔test", tr_te)
        c2.metric("Shared train↔validation", tr_va)
        c3.metric("Shared validation↔test", va_te)
        for w in d.split.warnings:
            st.warning(w)
    note("A grouped split shows **0** shared host stars. A stratified row split typically shares many, so its test score can be optimistic "
         "(objects around one star have correlated properties).")

with tabs[2]:
    counts = data.clean["label"].value_counts().reindex(data.class_names)
    c1, c2 = st.columns(2)
    with c1:
        show_fig(viz.fig_class_distribution(counts, "Class distribution after cleaning"), "class_distribution_clean", "prep_cls")
    with c2:
        majority = counts.max() / counts.sum()
        st.metric("Majority-class share", f"{majority:.1%}")
        st.metric("Imbalance ratio", f"{counts.max() / counts.min():.2f}")
        st.write(f"A classifier that always predicts the majority class would reach **{majority:.1%} accuracy** – "
                 "so Macro-F1, log loss, Brier score and calibration are reported alongside accuracy.")
        show_df(fmt_table(data.split_summary(), 3))

with tabs[3]:
    pre = data.preprocessor
    train_matrix = pre._to_matrix(data.train, pre.kept_features_)
    rows = []
    for i, f in enumerate(pre.kept_features_):
        raw = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append({
            "feature": f, "log10": f in pre.log_features,
            "missing_before_%": round(raw.isna().mean() * 100, 2),
            "raw_median": raw.median(), "raw_mean": raw.mean(), "raw_std": raw.std(),
            "imputed_cells_train": int(np.isnan(train_matrix[:, i]).sum()),
            "train_mean_after": data.X_train[:, i].mean(), "train_std_after": data.X_train[:, i].std(),
            "test_mean_after": data.X_test[:, i].mean(), "test_std_after": data.X_test[:, i].std(),
        })
    ba = pd.DataFrame(rows)
    show_df(fmt_table(ba, 3))
    st.caption("After scaling, the TRAINING mean/std are 0/1 by construction; validation/test deviate slightly because they use the "
               "training statistics (as they must).")
    download_table(ba, "preprocessing_before_after.csv", key="dl_ba")
    pick = st.selectbox("Inspect a feature", pre.kept_features_)
    i = pre.kept_features_.index(pick)
    raw_train = pd.to_numeric(data.train[pick], errors="coerce")
    show_fig(viz.fig_before_after(raw_train, train_matrix[:, i], data.X_train[:, i], pick, pick in pre.log_features),
             f"before_after_{pick}", "prep_ba")

with tabs[4]:
    st.markdown("**Features given to every model:** " + ", ".join(f"`{f}`" for f in data.features))
    med = pd.DataFrame({"feature": list(data.preprocessor.medians_), "training median (transformed scale)": list(data.preprocessor.medians_.values()),
                        "scaler mean": data.preprocessor.scaler_.mean_, "scaler std": data.preprocessor.scaler_.scale_})
    show_df(fmt_table(med, 4))
    st.caption("These objects are fitted on the training split only and are saved with the models.")
