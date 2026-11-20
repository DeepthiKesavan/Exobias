from __future__ import annotations

import pandas as pd


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    if out.columns.duplicated().any():
        seen: dict[str, int] = {}
        names = []
        for c in out.columns:
            seen[c] = seen.get(c, -1) + 1
            names.append(c if seen[c] == 0 else f"{c}.{seen[c]}")
        out.columns = names
    for col in out.columns:
        series = out[col]
        if series.dtype == object:
            out[col] = series.map(lambda v: v if v is None or (isinstance(v, float) and v != v) else str(v))
    if out.index.dtype == object:
        out.index = out.index.map(str)
    return out
