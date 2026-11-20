import numpy as np
import pandas as pd
import pytest

from src.table_utils import arrow_safe


def messy_frame():
    # mimics real NASA table quirks: text mixed with numbers, bools, bytes, None/NaN, duplicate-looking names
    return pd.DataFrame({
        "mixed": ["a", 1, 2.5, None, np.nan],
        "flags": [True, "n/a", 0, None, 1],
        "raw": [b"x", "y", 3, None, 4.0],
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "txt": ["u", "v", "w", "x", "y"],
    })


def test_object_columns_become_text_and_original_untouched():
    df = messy_frame()
    out = arrow_safe(df)
    for col in ("mixed", "flags", "raw", "txt"):
        assert all(isinstance(v, str) or v is None or v != v for v in out[col])
    assert out["num"].dtype == df["num"].dtype
    assert df["mixed"].iloc[1] == 1 and not isinstance(df["mixed"].iloc[1], str)  # original unchanged


def test_missing_values_stay_missing():
    out = arrow_safe(messy_frame())
    assert pd.isna(out["mixed"].iloc[3]) and pd.isna(out["mixed"].iloc[4])


def test_duplicate_and_non_string_column_names_are_made_unique():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", 5])
    out = arrow_safe(df)
    assert list(out.columns) == ["a", "a.1", "5"]


def test_pyarrow_can_convert_result_when_available():
    pa = pytest.importorskip("pyarrow")
    out = arrow_safe(messy_frame())
    assert pa.Table.from_pandas(out).num_rows == 5
