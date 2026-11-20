"""Plumbing tests for the ANN branch using a TEST DOUBLE in place of the Keras trainer.

These verify that experiments hand the right cohorts, sample weights and validation data to the
trainer, evaluate on the common test set, and persist artefacts. They do NOT test ANN learning;
that is covered by test_ann_trains_and_is_reproducible when TensorFlow is installed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src import ann_model
from src.ann_model import ANNConfig, ANNTrainingResult
from src.data_validation import DEFAULT_FEATURES
from src.experiments import run_all_strategies, save_experiment_artifacts
from src.preprocessing import prepare_data

CALLS = []


class DummyModel:
    def __init__(self, n_features, n_classes):
        self.w = np.random.default_rng(0).normal(size=(n_features, n_classes))

    def predict(self, X, verbose=0):
        z = X @ self.w
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path):
        Path(path).write_bytes(b"dummy")


def dummy_train(X_train, y_train, X_val, y_val, n_classes, config, sample_weight_train=None,
                sample_weight_val=None, progress_callback=None):
    CALLS.append({"n_train": len(y_train), "n_val": len(y_val), "w": sample_weight_train, "w_val": sample_weight_val})
    if progress_callback:
        progress_callback(1, config.max_epochs, {"val_loss": 0.6})
    hist = {"loss": [0.7, 0.5], "val_loss": [0.7, 0.6], "accuracy": [0.5, 0.7], "val_accuracy": [0.5, 0.65]}
    summary = {"best_epoch": 2, "epochs_run": 2, "neurons_per_hidden_layer": [64, 32]}
    return ANNTrainingResult(DummyModel(X_train.shape[1], n_classes), config, hist, 2, 2, X_train.shape[1], n_classes,
                             len(y_train), len(y_val), sample_weight_train is not None, 0.0, summary, ["dummy"])


@pytest.fixture()
def results(synthetic_koi, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(ann_model, "train_ann", dummy_train)
    data = prepare_data(synthetic_koi, DEFAULT_FEATURES, seed=2)
    res = run_all_strategies(data, include_ann=True, ann_config=ANNConfig(seed=2), baselines=["Logistic Regression"])
    return data, res


def test_trainer_receives_correct_cohorts_and_weights(results):
    data, res = results
    a, b, c = CALLS
    assert a["n_train"] == len(data.train) and a["w"] is None
    assert b["n_train"] < a["n_train"] and b["w"] is None
    assert c["w"] is not None and len(c["w"]) == c["n_train"] and np.isclose(c["w"].mean(), 1.0)
    assert c["w_val"] is not None and len(c["w_val"]) == c["n_val"]


def test_ann_evaluated_on_common_test_set(results):
    data, res = results
    for s in "ABC":
        ann = res[s].get("ANN")
        assert ann.proba_test.shape == (len(data.test), 2)
        assert ann.metrics["n_samples"] == len(data.test)
    assert res["B"].get("ANN").n_train < res["A"].get("ANN").n_train


def test_artifacts_saved_together(results, tmp_path):
    data, res = results
    info = save_experiment_artifacts(res, data, tmp_path, "t")
    folder = Path(info["directory"])
    assert (folder / "preprocessor.joblib").exists() and (folder / "experiment_config.json").exists()
    assert (folder / "A_ANN.keras").exists() and (folder / "C_Logistic_Regression.joblib").exists()
    meta = json.loads((folder / "experiment_config.json").read_text())
    assert meta["features"] == data.features and set(meta["experiments"]) == {"A", "B", "C"}
    assert meta["experiments"]["B"]["params"]["threshold"] == 0.8
