"""Shared test fixtures.

IMPORTANT: ``make_synthetic_koi`` generates SYNTHETIC rows that merely mimic the
column layout of the NASA KOI table so the code paths can be unit-tested offline.
It is test scaffolding only and must never be presented as NASA data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_synthetic_koi(n=1200, seed=0, with_confirmed=False, with_score=True, with_kepid=True):
    rng = np.random.default_rng(seed)
    n_fp = int(n * 0.5)
    labels = np.array(["FALSE POSITIVE"] * n_fp + ["CANDIDATE"] * (n - n_fp), dtype=object)
    if with_confirmed:
        cand_idx = np.flatnonzero(labels == "CANDIDATE")
        labels[rng.choice(cand_idx, size=len(cand_idx) // 3, replace=False)] = "CONFIRMED"
    is_planet = labels != "FALSE POSITIVE"
    period = np.where(is_planet, rng.lognormal(2.0, 1.2, n), rng.lognormal(1.2, 1.5, n))
    prad = np.where(is_planet, rng.lognormal(0.7, 0.6, n), rng.lognormal(1.5, 1.2, n))
    teq = rng.normal(900, 400, n).clip(100, 3500)
    insol = rng.lognormal(3.0, 1.8, n)
    steff = rng.normal(5600, 700, n).clip(3000, 9000)
    srad = rng.lognormal(0.0, 0.3, n)
    smass = rng.normal(1.0, 0.2, n).clip(0.3, 2.0)
    base = np.where(is_planet, rng.beta(5, 2, n), rng.beta(2, 5, n))
    df = pd.DataFrame({
        "kepoi_name": [f"K{i:05d}.01" for i in range(n)],
        "koi_disposition": labels,
        "koi_period": period, "koi_prad": prad, "koi_teq": teq, "koi_insol": insol,
        "koi_steff": steff, "koi_srad": srad, "koi_smass": smass,
        "koi_fpflag_nt": rng.integers(0, 2, n),
    })
    if with_score:
        df["koi_score"] = base
    if with_kepid:
        df["kepid"] = rng.integers(1_000_000, 1_000_000 + n // 2, n)  # some stars host several KOIs
    # realistic mess
    for col in ("koi_teq", "koi_insol", "koi_smass"):
        df.loc[rng.choice(n, size=n // 25, replace=False), col] = np.nan
    return df


@pytest.fixture(scope="session")
def synthetic_koi():
    return make_synthetic_koi()


@pytest.fixture(scope="session")
def synthetic_koi_three_class():
    return make_synthetic_koi(with_confirmed=True)
