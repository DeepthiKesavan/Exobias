import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import make_synthetic_koi
from src.data_validation import DEFAULT_FEATURES
from src.experiments import StrategyParams
from src.population_analysis import (
    ALL_KEY, CW_KEY, HC_KEY, compare_label_groups, compare_to_reference, effective_n,
    summarize_distribution, strategy_weight_vectors, weighted_correlation, weighted_ks,
    weighted_mean, weighted_quantile, weighted_std,
)
from src.preprocessing import clean_dataset


@pytest.fixture(scope="module")
def clean(synthetic_koi):
    return clean_dataset(synthetic_koi, DEFAULT_FEATURES)[0]


def test_unit_weights_match_numpy():
    rng = np.random.default_rng(0)
    v = rng.lognormal(size=501)
    w = np.ones_like(v)
    assert weighted_mean(v, w) == pytest.approx(v.mean())
    assert weighted_std(v, w) == pytest.approx(v.std())
    assert weighted_quantile(v, w, 0.5) == pytest.approx(np.median(v), rel=0.01)


def test_integer_weights_equal_repetition():
    v = np.array([1.0, 2.0, 10.0])
    w = np.array([1.0, 3.0, 1.0])
    rep = np.array([1.0, 2, 2, 2, 10.0])
    assert weighted_mean(v, w) == pytest.approx(rep.mean())
    assert weighted_quantile(v, w, 0.5) == pytest.approx(2.0)
    assert effective_n(w) == pytest.approx(25 / 11)


def test_weighted_ks_matches_scipy_for_unit_weights():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 400), rng.normal(0.4, 1, 500)
    d, p = weighted_ks(a, np.ones(400), b, np.ones(500))
    ref = stats.ks_2samp(a, b)
    assert d == pytest.approx(ref.statistic, abs=1e-9)
    assert abs(p - ref.pvalue) < 0.02  # asymptotic vs exact


def test_weight_vectors_structure(clean):
    wv = strategy_weight_vectors(clean, StrategyParams(threshold=0.8))
    assert set(wv) == {ALL_KEY, HC_KEY, CW_KEY}
    assert wv[ALL_KEY].sum() == len(clean)
    assert 0 < wv[HC_KEY].sum() < len(clean) and set(np.unique(wv[HC_KEY])) <= {0.0, 1.0}


def test_no_score_gives_only_all_records():
    c = clean_dataset(make_synthetic_koi().drop(columns=["koi_score"]), DEFAULT_FEATURES)[0]
    assert list(strategy_weight_vectors(c)) == [ALL_KEY]


def test_summary_ci_brackets_point_estimate(clean):
    s = summarize_distribution(clean.koi_prad, np.ones(len(clean)), n_boot=200, seed=1)
    assert s["median_ci_low"] <= s["median"] <= s["median_ci_high"]
    assert s["mean_ci_low"] <= s["mean"] <= s["mean_ci_high"] and s["n_rows"] == len(clean)


def test_bootstrap_is_seeded(clean):
    a = summarize_distribution(clean.koi_prad, np.ones(len(clean)), n_boot=100, seed=3)
    b = summarize_distribution(clean.koi_prad, np.ones(len(clean)), n_boot=100, seed=3)
    assert a == b


def test_compare_to_reference_detects_known_shift(clean):
    wv = strategy_weight_vectors(clean, StrategyParams(threshold=0.8))
    table = compare_to_reference(clean.koi_prad.to_numpy(), wv, n_boot=200, seed=0)
    assert len(table) == 2 and (table.n_compared <= table.n_reference).all()
    row = table[table.comparison.str.startswith(HC_KEY)].iloc[0]
    assert row.median_diff_ci_low <= row.median_diff <= row.median_diff_ci_high
    assert 0 <= row.ks_D <= 1


def test_identical_strategy_gives_zero_difference(clean):
    wv = {ALL_KEY: np.ones(len(clean)), "copy": np.ones(len(clean))}
    row = compare_to_reference(clean.koi_prad.to_numpy(), wv, n_boot=100).iloc[0]
    assert row.median_diff == 0 and row.ks_D == 0 and not row.median_diff_ci_excludes_0


def test_label_groups_differ_in_synthetic_radius(clean):
    wv = strategy_weight_vectors(clean)
    t = compare_label_groups(clean, "koi_prad", wv, n_boot=100)
    assert len(t) == 3 and (t.ks_p < 0.01).all()  # synthetic FP radii were generated larger


def test_weighted_correlation_recovers_monotone_relation():
    x = np.arange(200.0)
    assert weighted_correlation(x, x ** 2, np.ones(200)) == pytest.approx(1.0)
    assert weighted_correlation(x, -x, np.ones(200)) == pytest.approx(-1.0)
