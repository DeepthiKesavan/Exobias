"""Page: ANN Workbench - configure, train and inspect Experiments A / B / C."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_helpers import page_header

from src import visualizations as viz
from src.ann_model import ANNConfig, architecture_table, tensorflow_available
from src.app_helpers import (MODELS_DIR, SCORE_WARNING, fmt_table, init_state, note, require_data, setup_page,
                             show_cohort_info, show_fig, show_model_detail)
from src.baseline_models import available_baselines, xgboost_available
from src.experiments import STRATEGY_NAMES, StrategyParams, run_strategy, save_experiment_artifacts

setup_page("ANN Workbench")
init_state()
page_header("ANN Workbench", "Configure the ANN and traditional models, then train Experiments A, B and C on identical features and one common test set.", 3, "Modelling")
data = require_data()
report = st.session_state["validation"]
has_score = bool(report.has_score and data.train["score"].notna().any())
tf_ok = tensorflow_available()

st.markdown("### Experiments")
e1, e2, e3 = st.columns(3)
e1.info("**A · Baseline ANN**\n\nAll eligible rows. No score used. Reference model.")
e2.info("**B · High-confidence filtered**\n\nCandidates kept if score ≥ t, False Positives if score ≤ 1−t. "
        "Smaller training population; same original test set.")
e3.info("**C · Confidence-weighted**\n\nCandidate weight = score, False Positive weight = 1−score (training only, via Keras `sample_weight`).")
note(SCORE_WARNING, "warn")
if not has_score:
    st.warning("koi_score is unavailable (or entirely missing in training): only Experiment A can run.")

prev_params: StrategyParams = st.session_state.get("strategy_params") or StrategyParams()
prev_cfg: dict = st.session_state.get("ann_cfg_dict", {})

t_exp, t_arch, t_train, t_trad = st.tabs(["Experiment settings", "ANN architecture", "Training", "Traditional models"])

with t_exp:
    options = [s for s in "ABC" if s == "A" or has_score]
    chosen = st.multiselect("Experiments to train", options, default=options, format_func=lambda s: STRATEGY_NAMES[s])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Experiment B – filtering**")
        threshold = st.slider("Confidence threshold t", 0.50, 0.95, float(prev_params.threshold), 0.05,
                              help="Candidates need koi_score ≥ t; False Positives need koi_score ≤ 1 − t.",
                              disabled=not has_score)
        filter_val = st.checkbox("Also filter the validation set (used for early stopping)", value=prev_params.filter_validation,
                                 disabled=not has_score)
    with c2:
        st.markdown("**Experiment C – weighting**")
        min_w = st.slider("Minimum weight (floor)", 0.0, 0.5, float(prev_params.min_weight), 0.01, disabled=not has_score)
        exponents = [0.5, 1.0, 2.0]
        exponent = st.selectbox("Weight exponent (1.0 = proposed heuristic)", exponents,
                                index=exponents.index(prev_params.weight_exponent) if prev_params.weight_exponent in exponents else 1,
                                disabled=not has_score)
        policy = st.radio("Rows with a missing koi_score", ["exclude", "neutral"],
                          index=["exclude", "neutral"].index(prev_params.missing_score_policy), horizontal=True,
                          format_func=lambda p: "Exclude (default)" if p == "exclude" else "Neutral weight", disabled=not has_score)
        neutral_w = st.slider("Neutral weight", 0.05, 1.0, float(prev_params.neutral_weight), 0.05,
                              disabled=not has_score or policy != "neutral")
    st.caption("Weights are rescaled so their training mean is 1.0, keeping the effective learning rate comparable. "
               "They are used only during training; all test metrics are ordinary (unweighted).")

with t_arch:
    c1, c2 = st.columns(2)
    n_hidden = int(c1.number_input("Hidden layers", 1, 4, value=len(prev_cfg.get("hidden_units", (64, 32)))))
    defaults = list(prev_cfg.get("hidden_units", (64, 32))) + [16, 8, 8, 8]
    units = [int(c2.number_input(f"Neurons – hidden layer {i + 1}", 2, 512, value=int(defaults[i]), key=f"units_{i}"))
             for i in range(n_hidden)]
    c3, c4 = st.columns(2)
    activations = ["relu", "tanh", "elu"]
    activation = c3.selectbox("Hidden activation", activations,
                              index=activations.index(prev_cfg.get("activation", "relu")))
    dropout = c4.slider("Dropout rate", 0.0, 0.6, float(prev_cfg.get("dropout", 0.20)), 0.05)

with t_train:
    c1, c2, c3 = st.columns(3)
    lr = c1.number_input("Learning rate (Adam)", 0.00001, 0.1, float(prev_cfg.get("learning_rate", 0.001)), 0.0001, format="%.5f")
    batches = [16, 32, 64, 128]
    batch = c2.selectbox("Batch size", batches, index=batches.index(prev_cfg.get("batch_size", 32)))
    max_epochs = int(c3.number_input("Maximum epochs", 1, 500, int(prev_cfg.get("max_epochs", 100))))
    c4, c5 = st.columns(2)
    patience = int(c4.number_input("Early-stopping patience (val loss)", 1, 100, int(prev_cfg.get("patience", 12))))
    ann_seed = int(c5.number_input("ANN random seed", 0, 10_000, int(prev_cfg.get("seed", data.seed))))
    st.caption("Optimizer Adam · loss sparse categorical cross-entropy · best weights restored on early stop.")

with t_trad:
    pool = available_baselines()
    chosen_baselines = st.multiselect("Traditional models trained on the same cohorts", pool,
                                      default=[m for m in ("Logistic Regression", "Random Forest") if m in pool])
    if not xgboost_available():
        st.caption("XGBoost is optional and not installed (`pip install xgboost`).")
    include_ann = st.checkbox("Train the ANN", value=tf_ok, disabled=not tf_ok)
    if not tf_ok:
        st.error("TensorFlow is not installed, so the ANN cannot be trained here. Install it with `pip install tensorflow` "
                 "(Python 3.10–3.12 recommended). Traditional models can still be run.")

# ---- configuration summary (works without TensorFlow) ------------------------------------
try:
    ann_cfg = ANNConfig(hidden_units=tuple(units), activation=activation, dropout=dropout, learning_rate=float(lr),
                        batch_size=int(batch), max_epochs=max_epochs, patience=patience, seed=ann_seed)
    params = StrategyParams(threshold=float(threshold), min_weight=float(min_w), weight_exponent=float(exponent),
                            missing_score_policy=policy, neutral_weight=float(neutral_w), filter_validation=bool(filter_val)).validate()
except ValueError as exc:
    st.error(f"Invalid configuration: {exc}")
    st.stop()

st.markdown("### Architecture summary")
arch = architecture_table(len(data.features), data.n_classes, ann_cfg)
c1, c2 = st.columns([3, 2])
c1.dataframe(arch)
c2.markdown(
    f"""
* **Input features:** {len(data.features)} ({', '.join(data.features)})
* **Output neurons:** {data.n_classes} (softmax) → {', '.join(data.class_names)}
* **Hidden layers:** {len(units)} · **neurons:** {units}
* **Activation:** {activation} → softmax · **Dropout:** {dropout}
* **Optimizer:** Adam · **learning rate:** {lr:g} · **batch size:** {batch}
* **Max epochs:** {max_epochs} · **patience:** {patience}
* **Trainable parameters:** {int(arch['parameters'].sum()):,}
"""
)

# ---- training -----------------------------------------------------------------------------
st.markdown("### Train")
if st.button("Train selected experiments", type="primary", disabled=not chosen):
    st.session_state["strategy_params"] = params
    st.session_state["ann_cfg_dict"] = ann_cfg.to_dict()
    bar = st.progress(0.0, text="Starting…")
    done_total = len(chosen)
    for idx, s in enumerate(chosen):
        def make_cb(strategy=s, index=idx):
            def cb(epoch, total_epochs, logs):
                frac = (index + epoch / total_epochs) / done_total
                bar.progress(min(frac, 0.99), text=f"Experiment {strategy}: epoch {epoch}/{total_epochs} "
                                                    f"· val_loss {logs.get('val_loss', float('nan')):.4f}")
            return cb
        try:
            bar.progress(idx / done_total, text=f"Experiment {s}: preparing cohort and training…")
            result = run_strategy(data, s, params, ann_cfg, include_ann=include_ann, baselines=chosen_baselines,
                                  progress_callback=make_cb())
            st.session_state["results"][s] = result
        except Exception as exc:
            st.error(f"Experiment {s} failed: {exc}")
    bar.progress(1.0, text="Finished.")
    st.session_state["saved_models"] = None

results = st.session_state.get("results", {})
if not results:
    st.info("Nothing trained yet. Training starts only when you press the button.")
    st.stop()

# ---- save ---------------------------------------------------------------------------------
c1, c2 = st.columns([1, 3])
if c1.button("Save models + preprocessor"):
    try:
        tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        st.session_state["saved_models"] = save_experiment_artifacts(results, data, MODELS_DIR, tag)
    except Exception as exc:
        st.error(f"Saving failed: {exc}")
saved = st.session_state.get("saved_models")
if saved:
    c2.success(f"Saved to `{saved['directory']}` (models, preprocessor.joblib, feature names and experiment_config.json).")

# ---- results ------------------------------------------------------------------------------
st.markdown("### Results")
tabs = st.tabs([STRATEGY_NAMES[s] for s in results])
for tab, (s, exp) in zip(tabs, results.items()):
    with tab:
        st.markdown("**Training cohort**")
        show_cohort_info(exp.cohort_info)
        ann = exp.get("ANN")
        if ann is not None:
            a = ann.ann_summary
            st.markdown("**ANN training summary**")
            cols = st.columns(4)
            cols[0].metric("Best epoch", f"{a['best_epoch']} / {a['epochs_run']}")
            cols[1].metric("Train loss (best)", f"{a['train_loss_at_best']:.4f}")
            cols[2].metric("Val loss (best)", f"{a['val_loss_at_best']:.4f}")
            cols[3].metric("Parameters", f"{a['total_parameters']:,}")
            cols = st.columns(4)
            cols[0].metric("Train accuracy", f"{a['train_accuracy_at_best']:.4f}")
            cols[1].metric("Val accuracy", f"{a['val_accuracy_at_best']:.4f}")
            cols[2].metric("Epochs run", a["epochs_run"])
            cols[3].metric("Train time (s)", f"{ann.train_seconds:.1f}")
            if a.get("sample_weighted"):
                st.caption("Validation loss/accuracy in this run are sample-weighted (same weights heuristic).")
            show_fig(viz.fig_learning_curves(ann.history, a["best_epoch"], f"ANN – {STRATEGY_NAMES[s]}"), f"learning_{s}", f"lc_{s}")
            for obs in ann.observations:
                st.write("• " + obs)
        names = [m.model_name for m in exp.models]
        pick = st.selectbox("Model detail", names, key=f"pick_{s}")
        show_model_detail(exp.get(pick), data, f"wb_{s}_{pick}")
