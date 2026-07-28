"""Synthetic 3-factor yield generator.

Purpose: (1) unit tests (does the engine recover a known 3-factor structure?),
(2) offline smoke-test of the FULL pipeline (`run_all --synthetic`) without a
FRED key. Crisis windows get 4x level-factor volatility so the rolling plot
shows the expected PC1-EVR spikes.

This is a plumbing tool, NOT data for the README: all reported results must
come from real FRED data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CRISES = [("2008-01-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]


def factor_loadings() -> np.ndarray:
    """Stylized level/slope/curvature loadings over the 8 tenors."""
    m = np.log(np.array(config.MATURITIES))
    m = (m - m.min()) / (m.max() - m.min())          # 0..1
    level = np.ones(8)
    slope = (m - 0.5) * 2
    curv = 1 - np.abs(m - 0.5) * 4
    L = np.stack([level, slope, curv], axis=1)
    return L / np.linalg.norm(L, axis=0)


def make_synthetic(seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(config.START_DATE, "2026-06-30")
    T = len(idx)
    L = factor_loadings()
    vol = np.array([0.05, 0.02, 0.008])              # pp/day
    f = rng.standard_normal((T, 3)) * vol
    crisis_mask = np.zeros(T, dtype=bool)
    for s, e in CRISES:
        crisis_mask |= (idx >= s) & (idx <= e)
    f[crisis_mask, 0] *= 4.0                          # level vol regime
    noise = rng.standard_normal((T, 8)) * 0.004
    changes = f @ L.T + noise
    base = 3.0 + 0.8 * np.log1p(np.array(config.MATURITIES))
    levels = pd.DataFrame(base + np.cumsum(changes, axis=0),
                          index=idx, columns=config.COLUMNS)
    levels.index.name = "date"
    usrec = pd.Series(0, index=idx, name="usrec")
    for s, e in [("2008-01-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]:
        usrec.loc[s:e] = 1
    return levels, usrec
