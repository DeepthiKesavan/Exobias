import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_koi
from src.data_validation import DEFAULT_FEATURES
from src.preprocessing import FeaturePreprocessor, clean_dataset, prepare_data, split_dataset


def test_cleaning_log_accounts_for_every_row(synthetic_koi):
    df = pd.concat([synthetic_koi, synthetic_koi.iloc[:10]], ignore_index=True)  # 10 duplicates
    df.loc[0, "koi_disposition"] = "NOT DISPOSITIONED"
    df.loc[1, "koi_period"] = np.inf
    clean, rep = clean_dataset(df, DEFAULT_FEATURES)
    log = rep.removal_log
    assert log.iloc[0].remaining == len(df)
    assert log.removed.sum() == len(df) - len(clean)
    # 10 rows were copied, but original row 0 was already dropped by the label filter,
    # so its (correctly labelled) copy survives as the first occurrence -> 9 duplicates.
    assert log.loc[log.stage == "Duplicate records", "removed"].iloc[0] == 9
    assert rep.inf_replaced >= 1
    assert np.isfinite(clean[DEFAULT_FEATURES].to_numpy(dtype=float)[~np.isnan(clean[DEFAULT_FEATURES].to_numpy(dtype=float))]).all()


def test_clean_rejects_leaky_feature(synthetic_koi):
    with pytest.raises(ValueError):
        clean_dataset(synthetic_koi, ["koi_period", "koi_score"])


def test_grouped_split_has_no_star_overlap(synthetic_koi):
    clean, _ = clean_dataset(synthetic_koi, DEFAULT_FEATURES)
    sp = split_dataset(clean, "grouped", seed=1)
    assert sp.mode_used == "grouped"
    g = [set(clean.group.iloc[i]) for i in (sp.train_idx, sp.val_idx, sp.test_idx)]
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
    n = len(clean)
    assert abs(len(sp.train_idx) / n - 0.70) < 0.06
    assert abs(len(sp.test_idx) / n - 0.15) < 0.06
    assert len(sp.train_idx) + len(sp.val_idx) + len(sp.test_idx) == n


def test_stratified_split_preserves_balance(synthetic_koi):
    clean, _ = clean_dataset(synthetic_koi, DEFAULT_FEATURES)
    sp = split_dataset(clean, "stratified", seed=1)
    full = (clean.label == "CANDIDATE").mean()
    for idx in (sp.train_idx, sp.val_idx, sp.test_idx):
        assert abs((clean.label.iloc[idx] == "CANDIDATE").mean() - full) < 0.03


def test_grouped_falls_back_without_kepid_with_warning():
    df = make_synthetic_koi().drop(columns=["kepid"])
    clean, _ = clean_dataset(df, DEFAULT_FEATURES)
    sp = split_dataset(clean, "grouped")
    assert sp.mode_used == "stratified" and sp.warnings


def test_split_is_reproducible(synthetic_koi):
    clean, _ = clean_dataset(synthetic_koi, DEFAULT_FEATURES)
    a, b = split_dataset(clean, "grouped", seed=7), split_dataset(clean, "grouped", seed=7)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_preprocessor_fits_on_train_only(synthetic_koi):
    data = prepare_data(synthetic_koi, DEFAULT_FEATURES, seed=3)
    pre = data.preprocessor
    # scaler statistics must equal those computed from the training matrix alone
    train_matrix = pre.imputer_.transform(pre._to_matrix(data.train, pre.kept_features_))
    assert np.allclose(pre.scaler_.mean_, train_matrix.mean(axis=0), atol=1e-6)
    assert np.allclose(data.X_train.mean(axis=0), 0, atol=1e-4)
    # val/test are NOT re-centred on their own statistics
    assert not np.allclose(data.X_test.mean(axis=0), 0, atol=1e-6)
    assert not np.isnan(data.X_test).any() and not np.isnan(data.X_val).any()


def test_imputer_uses_training_median_for_test_rows(synthetic_koi):
    data = prepare_data(synthetic_koi, DEFAULT_FEATURES, seed=3)
    probe = data.test.iloc[[0]].copy()
    probe["koi_teq"] = np.nan
    imputed = data.preprocessor.transform_imputed(probe)
    col = data.features.index("koi_teq")
    assert np.isclose(imputed[0, col], data.preprocessor.medians_["koi_teq"], rtol=1e-5)


def test_prepare_data_label_encoding(synthetic_koi, synthetic_koi_three_class):
    d2 = prepare_data(synthetic_koi, DEFAULT_FEATURES)
    assert d2.class_names == ["FALSE POSITIVE", "CANDIDATE"] and set(d2.y_train) == {0, 1}
    d3 = prepare_data(synthetic_koi_three_class, DEFAULT_FEATURES)
    assert d3.n_classes == 3 and set(d3.y_test) == {0, 1, 2}


def test_preprocessor_save_load_roundtrip(tmp_path, synthetic_koi):
    data = prepare_data(synthetic_koi, DEFAULT_FEATURES)
    path = tmp_path / "pre.joblib"
    data.preprocessor.save(path)
    loaded = FeaturePreprocessor.load(path)
    assert np.allclose(loaded.transform(data.test), data.X_test)
