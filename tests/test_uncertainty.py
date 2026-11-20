import numpy as np
import pytest

from conftest import make_synthetic_koi
from src.data_validation import DEFAULT_FEATURES
from src.preprocessing import prepare_data
from src.uncertainty_analysis import (
    class_distribution_comparison, dataset_sensitivity, feature_sensitivity, multi_seed_experiment,
    population_stability, split_comparison, stability_assessment, summarise_across_seeds,
    threshold_sweep, weight_sensitivity,
)

LIGHT = ["Logistic Regression"]


@pytest.fixture(scope="module")
def df():
    return make_synthetic_koi(n=900, seed=4)


def test_multi_seed_and_summary(df):
    t = multi_seed_experiment(df, DEFAULT_FEATURES, [1, 2, 3], baselines=LIGHT)
    assert set(t.seed) == {1, 2, 3} and set(t.strategy_key) == {"A", "B", "C"}
    s = summarise_across_seeds(t)
    assert (s.n_runs == 3).all() and (s.f1_macro_std >= 0).all()
    # different seeds really give different splits -> some variation
    assert t.groupby("strategy_key").accuracy.std().max() > 0


def test_stability_verdicts(df):
    t = multi_seed_experiment(df, DEFAULT_FEATURES, [1, 2, 3, 4], baselines=["Logistic Regression", "Random Forest"])
    out = stability_assessment(t)
    assert len(out) and out.verdict.isin(
        {"Stable improvement", "Stable decline", "No reliable difference", "Identical in every seed"}).all()
    short = stability_assessment(t[t.seed <= 2])
    assert (short.verdict == "Insufficient seeds (need at least 3)").all()


def test_threshold_sweep_monotone_training_size(df):
    data = prepare_data(df, DEFAULT_FEATURES, seed=1)
    t = threshold_sweep(data, (0.7, 0.8, 0.9), baselines=LIGHT)
    sizes = t[(t.test_subset == "full test set") & t.threshold.notna()].sort_values("threshold").n_train.tolist()
    assert sizes == sorted(sizes, reverse=True)
    hc = t[t.test_subset.str.startswith("high-confidence")]
    assert hc.n_samples.nunique() == 1  # one common reference subset for every model


def test_weight_sensitivity_runs_all_settings(df):
    data = prepare_data(df, DEFAULT_FEATURES, seed=1)
    t = weight_sensitivity(data, baselines=LIGHT)
    assert t.weight_setting.nunique() == 6
    neutral = t[t.weight_setting.str.contains("neutral")].n_train.iloc[0]
    assert neutral >= t[t.weight_setting.str.startswith("score (default")].n_train.iloc[0]


def test_split_comparison_and_features(df):
    s = split_comparison(df, DEFAULT_FEATURES, [1, 2], baselines=LIGHT, strategies=("A",))
    assert set(s.split_requested) == {"grouped", "stratified"}
    f = feature_sensitivity(df, DEFAULT_FEATURES, baselines=LIGHT)
    assert "planetary features only" in set(f.feature_set) and "without koi_prad" in set(f.feature_set)
    assert f.loc[f.feature_set == "all features", "f1_macro_change_vs_all"].eq(0).all()


def test_class_distribution_and_population_stability(df):
    data = prepare_data(df, DEFAULT_FEATURES, seed=1)
    c = class_distribution_comparison(data)
    assert c.rows.iloc[0] == len(data.train) and any("training cohort" in p for p in c.population)
    p = population_stability(df, DEFAULT_FEATURES, n_boot=50)
    assert p.setting.iloc[0] == "All eligible records" and len(p) >= 4


def test_dataset_sensitivity_handles_different_sources(df):
    other = make_synthetic_koi(n=700, seed=9, with_score=False)
    t = dataset_sensitivity({"v1": df, "v2_no_score": other}, DEFAULT_FEATURES, baselines=LIGHT)
    assert set(t[t.dataset == "v2_no_score"].strategy_key) == {"A"}
    assert set(t[t.dataset == "v1"].strategy_key) == {"A", "B", "C"}
