import io

import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_koi
from src.data_loader import load_koi_csv
from src.data_validation import (
    assert_features_allowed, candidate_feature_columns, class_counts, is_forbidden_feature,
    normalise_label, validate_koi_dataframe,
)


def test_normalise_label_variants():
    assert normalise_label("false positive") == "FALSE POSITIVE"
    assert normalise_label("FALSE_POSITIVE") == "FALSE POSITIVE"
    assert normalise_label(" Candidate ") == "CANDIDATE"
    assert normalise_label("NOT DISPOSITIONED") is None
    assert normalise_label(np.nan) is None


def test_forbidden_features_blocked():
    for bad in ["koi_disposition", "koi_score", "kepid", "koi_fpflag_nt", "koi_fpflag_ss",
                "koi_fpflag_co", "koi_fpflag_ec", "koi_pdisposition"]:
        assert is_forbidden_feature(bad)
    with pytest.raises(ValueError):
        assert_features_allowed(["koi_period", "koi_score"])
    assert_features_allowed(["koi_period", "koi_prad"])


def test_candidate_feature_columns_exclude_leaky(synthetic_koi):
    cols = candidate_feature_columns(synthetic_koi)
    assert "koi_period" in cols
    assert "koi_score" not in cols and "kepid" not in cols and "koi_fpflag_nt" not in cols


def test_binary_dataset_reports_no_confirmed(synthetic_koi):
    rep = validate_koi_dataframe(synthetic_koi)
    assert rep.is_trainable
    assert rep.supported_classes == ["FALSE POSITIVE", "CANDIDATE"]
    assert not rep.has_confirmed and not rep.supports_three_class
    assert rep.has_score and rep.has_group


def test_three_class_detected(synthetic_koi_three_class):
    rep = validate_koi_dataframe(synthetic_koi_three_class)
    assert rep.supports_three_class


def test_missing_target_is_error():
    rep = validate_koi_dataframe(make_synthetic_koi().drop(columns=["koi_disposition"]))
    assert not rep.is_trainable


def test_missing_score_and_kepid_warn_not_fail():
    df = make_synthetic_koi().drop(columns=["koi_score", "kepid"])
    rep = validate_koi_dataframe(df)
    assert rep.is_trainable and not rep.has_score and not rep.has_group
    assert any("koi_score" in w for w in rep.warnings)


def test_tiny_class_is_error():
    df = make_synthetic_koi(n=200)
    df.loc[df.index[df.koi_disposition == "CANDIDATE"][5:], "koi_disposition"] = "FALSE POSITIVE"
    assert not validate_koi_dataframe(df).is_trainable


def test_class_counts_include_unsupported():
    df = make_synthetic_koi(n=100)
    df.loc[:4, "koi_disposition"] = "NOT DISPOSITIONED"
    assert class_counts(df)["UNSUPPORTED / MISSING"] == 5


def test_loader_skips_nasa_comment_header():
    text = "# This file was produced by the NASA Exoplanet Archive\n# COLUMN koi_period\nkepid,koi_disposition,koi_period\n1,CANDIDATE,3.2\n2,FALSE POSITIVE,1.1\n"
    df = load_koi_csv(io.BytesIO(text.encode()))
    assert list(df.columns) == ["kepid", "koi_disposition", "koi_period"] and len(df) == 2


@pytest.mark.parametrize("payload", [b"", b"   \n", b"# only comments\n# nothing else\n"])
def test_loader_rejects_empty(payload):
    with pytest.raises(ValueError):
        load_koi_csv(io.BytesIO(payload))
