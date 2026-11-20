import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_koi
from src.ann_model import ANNConfig, architecture_table, tensorflow_available
from src.data_validation import DEFAULT_FEATURES
from src.evaluation import (
    compute_metrics, expected_calibration_error, multiclass_brier, overfitting_observations,
)
from src.experiments import (
    StrategyParams, build_cohort, confidence_weights, high_confidence_mask, results_to_frame,
    run_all_strategies, run_strategy,
)
from src.preprocessing import prepare_data


@pytest.fixture(scope="module")
def data(synthetic_koi):
    return prepare_data(synthetic_koi, DEFAULT_FEATURES, seed=5)


# ---------------------------------------------------------------- masks and weights
def test_high_confidence_rule():
    labels = pd.Series(["CANDIDATE", "CANDIDATE", "FALSE POSITIVE", "FALSE POSITIVE", "CANDIDATE", "FALSE POSITIVE"])
    scores = pd.Series([0.85, 0.79, 0.15, 0.30, np.nan, np.nan])
    keep = high_confidence_mask(labels, scores, 0.80)
    assert keep.tolist() == [True, False, True, False, False, False]


def test_threshold_boundaries_inclusive():
    labels = pd.Series(["CANDIDATE", "FALSE POSITIVE"])
    assert high_confidence_mask(labels, pd.Series([0.8, 0.2]), 0.8).all()


def test_weights_follow_heuristic_and_floor():
    labels = pd.Series(["CANDIDATE", "FALSE POSITIVE", "CANDIDATE", "FALSE POSITIVE"])
    scores = pd.Series([0.9, 0.1, 0.02, 0.99])
    w = confidence_weights(labels, scores, StrategyParams(min_weight=0.05))
    assert np.allclose(w, [0.9, 0.9, 0.05, 0.05])


def test_missing_score_policies():
    labels = pd.Series(["CANDIDATE", "FALSE POSITIVE"])
    scores = pd.Series([np.nan, 0.2])
    excl = confidence_weights(labels, scores, StrategyParams(missing_score_policy="exclude"))
    neut = confidence_weights(labels, scores, StrategyParams(missing_score_policy="neutral", neutral_weight=0.5))
    assert np.isnan(excl[0]) and neut[0] == 0.5


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StrategyParams(threshold=0.3).validate()
    with pytest.raises(ValueError):
        StrategyParams(missing_score_policy="guess").validate()


# ---------------------------------------------------------------- cohorts
def test_cohort_a_uses_everything(data):
    c = build_cohort(data, "A", StrategyParams())
    assert c.info["n_train"] == len(data.train) and c.w_train is None and c.info["n_train_removed"] == 0


def test_cohort_b_filters_and_reports_removals(data):
    c = build_cohort(data, "B", StrategyParams(threshold=0.8))
    assert c.info["n_train"] < len(data.train)
    assert sum(c.info["removed_by_class"].values()) == c.info["n_train_removed"]
    keep = high_confidence_mask(data.train.label, data.train.score, 0.8)
    assert len(c.y_train) == keep.sum() and np.array_equal(c.y_train, data.y_train[keep])


def test_higher_threshold_keeps_fewer_rows(data):
    sizes = [build_cohort(data, "B", StrategyParams(threshold=t)).info["n_train"] for t in (0.7, 0.8, 0.9)]
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_cohort_c_weights_normalised_on_training_mean(data):
    c = build_cohort(data, "C", StrategyParams())
    assert np.isclose(c.w_train.mean(), 1.0) and (c.w_train > 0).all()
    assert len(c.w_train) == len(c.y_train) and len(c.w_val) == len(c.y_val)


def test_cohort_c_excludes_missing_scores_by_default():
    df = make_synthetic_koi(seed=2)
    df.loc[df.sample(150, random_state=1).index, "koi_score"] = np.nan
    d = prepare_data(df, DEFAULT_FEATURES, seed=2)
    excl = build_cohort(d, "C", StrategyParams(missing_score_policy="exclude"))
    neut = build_cohort(d, "C", StrategyParams(missing_score_policy="neutral"))
    assert excl.info["n_train"] < len(d.train) and neut.info["n_train"] == len(d.train)


def test_b_and_c_fail_cleanly_without_score():
    d = prepare_data(make_synthetic_koi().drop(columns=["koi_score"]), DEFAULT_FEATURES)
    assert build_cohort(d, "A", StrategyParams()).info["n_train"] == len(d.train)
    for s in "BC":
        with pytest.raises(ValueError):
            build_cohort(d, s, StrategyParams())


def test_features_never_include_score_or_flags(data):
    assert not any(f.startswith(("koi_fpflag", "koi_score", "kepid")) for f in data.features)
    assert data.X_train.shape[1] == len(data.features)


# ---------------------------------------------------------------- end to end (no ANN)
def test_run_all_strategies_traditional_only(data):
    res = run_all_strategies(data, include_ann=False, baselines=["Logistic Regression", "Random Forest"])
    table = results_to_frame(res)
    assert set(table.strategy_key) == {"A", "B", "C"} and len(table) == 6
    for col in ("accuracy", "f1_macro", "log_loss", "brier", "ece", "roc_auc"):
        assert table[col].between(0, 1.0 + 1e-9).all() or col == "log_loss"
    # All models share the same common test set
    n_test = {m.metrics["n_samples"] for e in res.values() for m in e.models}
    assert n_test == {len(data.test)}
    # filtered model really has the smaller training set
    sizes = table.groupby("strategy_key").n_train.first()
    assert sizes["B"] < sizes["A"]
    hc = results_to_frame(res, subset="hc")
    assert (hc.n_samples < len(data.test)).all()


def test_random_forest_beats_chance_on_separable_synthetic(data):
    r = run_strategy(data, "A", include_ann=False, baselines=["Random Forest"])
    assert r.models[0].metrics["roc_auc"] > 0.7


def test_three_class_pipeline(synthetic_koi_three_class):
    d = prepare_data(synthetic_koi_three_class, DEFAULT_FEATURES, seed=1)
    r = run_strategy(d, "A", include_ann=False, baselines=["Logistic Regression", "Random Forest"])
    assert r.models[0].proba_test.shape[1] == 3
    assert np.isnan(r.models[0].metrics["f1_pos"])


# ---------------------------------------------------------------- metrics
def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    assert expected_calibration_error(y, perfect)[0] == pytest.approx(0.0)
    wrong = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(y, wrong)[0] == pytest.approx(1.0)


def test_brier_definitions():
    y = np.array([1, 0])
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    m = compute_metrics(y, proba, 2)
    assert m["brier"] == pytest.approx(np.mean([(0.8 - 1) ** 2, (0.1 - 0) ** 2]))
    assert multiclass_brier(y, proba) == pytest.approx(2 * m["brier"])


def test_metrics_handle_single_class_gracefully():
    m = compute_metrics(np.array([1, 1, 1]), np.array([[0.2, 0.8]] * 3), 2)
    assert np.isnan(m["roc_auc"]) and 0 <= m["accuracy"] <= 1


def test_overfitting_notes_generated():
    hist = {"loss": [0.6, 0.4, 0.3, 0.25, 0.2], "val_loss": [0.6, 0.45, 0.46, 0.5, 0.55]}
    notes = overfitting_observations(hist, best_epoch=2, patience=2)
    assert any("early" in n.lower() for n in notes) and any("rose" in n for n in notes)


# ---------------------------------------------------------------- ANN
def test_architecture_table_matches_spec():
    t = architecture_table(7, 2, ANNConfig())
    assert list(t.layer) == ["Input", "Hidden 1", "Dropout 1", "Hidden 2", "Output"]
    assert t.parameters.sum() == (7 * 64 + 64) + (64 * 32 + 32) + (32 * 2 + 2)


def test_ann_config_validation():
    with pytest.raises(ValueError):
        ANNConfig(hidden_units=())
    with pytest.raises(ValueError):
        ANNConfig(dropout=1.0)


@pytest.mark.skipif(not tensorflow_available(), reason="TensorFlow not installed")
def test_ann_trains_and_is_reproducible(data):
    cfg = ANNConfig(max_epochs=4, patience=2, seed=11)
    a = run_strategy(data, "C", ann_config=cfg, baselines=[])
    b = run_strategy(data, "C", ann_config=cfg, baselines=[])
    ann_a, ann_b = a.models[0], b.models[0]
    assert ann_a.history and "val_loss" in ann_a.history
    assert ann_a.proba_test.shape == (len(data.test), 2)
    assert np.allclose(ann_a.proba_test.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(ann_a.proba_test, ann_b.proba_test, atol=1e-4)
