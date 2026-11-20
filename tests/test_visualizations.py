"""Smoke tests: every figure builder must render to a non-empty PNG on realistic inputs."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_koi
from src import visualizations as viz
from src.data_validation import DEFAULT_FEATURES
from src.evaluation import confusion_matrix_df
from src.experiments import run_all_strategies, results_to_frame
from src.population_analysis import (
    binned_median, compare_to_reference, correlation_table, prediction_based_summary,
    strategy_weight_vectors, summary_table,
)
from src.preprocessing import prepare_data
from src.uncertainty_analysis import multi_seed_experiment, stability_assessment, summarise_across_seeds, threshold_sweep


@pytest.fixture(scope="module")
def setup():
    df = make_synthetic_koi(n=800, seed=3)
    data = prepare_data(df, DEFAULT_FEATURES, seed=1)
    res = run_all_strategies(data, include_ann=False, baselines=["Logistic Regression", "Random Forest"])
    return df, data, res


def _ok(fig):
    png = viz.fig_to_png(fig, dpi=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_exploration_figures(setup):
    df, data, _ = setup
    _ok(viz.fig_class_distribution(data.clean["label"].value_counts()))
    _ok(viz.fig_missing_values(data.cleaning.missing_before, len(df)))
    _ok(viz.fig_feature_distributions(data.clean, DEFAULT_FEATURES, log_features=data.log_features))
    _ok(viz.fig_correlation_heatmap(data.clean, DEFAULT_FEATURES, data.log_features))


def test_ann_and_comparison_figures(setup):
    _, data, res = setup
    hist = {"loss": [0.7, 0.5, 0.4], "val_loss": [0.7, 0.55, 0.5], "accuracy": [0.5, 0.7, 0.8], "val_accuracy": [0.5, 0.65, 0.7]}
    _ok(viz.fig_learning_curves(hist, best_epoch=3))
    m = res["A"].models[0]
    _ok(viz.fig_confusion_matrix(confusion_matrix_df(data.y_test, m.proba_test, data.class_names)))
    table = results_to_frame(res)
    for metric in ("accuracy", "f1_macro", "ece"):
        _ok(viz.fig_metric_comparison(table, metric))
    _ok(viz.fig_roc(res, data.y_test))
    _ok(viz.fig_pr(res, data.y_test))
    _ok(viz.fig_reliability(res, data.y_test))


def test_population_figures(setup):
    _, data, res = setup
    clean = data.clean
    wv = strategy_weight_vectors(clean)
    _ok(viz.fig_distribution_overlay(clean.koi_prad, wv, "Planet radius (R_earth)"))
    _ok(viz.fig_box_by_strategy(clean.koi_prad, wv, "Planet radius (R_earth)"))
    _ok(viz.fig_ci_forest(summary_table(clean.koi_prad, wv, n_boot=50)))
    _ok(viz.fig_period_radius_scatter(clean))
    trends = {k: binned_median(clean.koi_steff, clean.koi_prad, w, np.linspace(4000, 7000, 7)) for k, w in wv.items()}
    _ok(viz.fig_binned_trend(trends, "Teff", "radius"))
    ct = correlation_table(clean, [("koi_steff", "koi_prad"), ("koi_srad", "koi_prad")], wv, n_boot=30)
    _ok(viz.fig_correlation_forest(ct))
    pred = prediction_based_summary(data.test, {"RF": res["A"].models[1].proba_test}, data.class_names, n_boot=30)
    _ok(viz.fig_prediction_summary(pred))


def test_uncertainty_figures(setup):
    df, data, _ = setup
    t = multi_seed_experiment(df, DEFAULT_FEATURES, [1, 2, 3], baselines=["Logistic Regression", "Random Forest"])
    _ok(viz.fig_seed_stability(summarise_across_seeds(t)))
    _ok(viz.fig_stability_forest(stability_assessment(t)))
    sweep = threshold_sweep(data, (0.7, 0.8, 0.9), baselines=["Logistic Regression"])
    sub = sweep[(sweep.test_subset == "full test set") & sweep.threshold.notna()]
    _ok(viz.fig_sweep_lines(sub, "threshold", "f1_macro"))
    _ok(viz.fig_grouped_bars(sub, "model", "f1_macro", "threshold"))
