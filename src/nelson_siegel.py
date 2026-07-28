"""Nelson-Siegel / Diebold-Li (2006) parametric term structure model, and the
bridge to the statistical PCA factors.

PCA finds level/slope/curvature from the data, assuming no functional form.
Nelson-Siegel assumes the three shapes up front and fits coefficients. The two
methods share nothing mechanically, so agreement between them is evidence the
structure is a property of the curve rather than of either method.

    y(tau) = b0
           + b1 * [(1 - exp(-L*tau)) / (L*tau)]
           + b2 * [(1 - exp(-L*tau)) / (L*tau) - exp(-L*tau)]

b0 is level (loading 1 everywhere), b1 slope (1 at the short end, decaying),
b2 curvature (0 at both ends, humped in between). Lambda is fixed rather than
fitted daily, which is Diebold-Li's contribution: it makes each day an OLS
solve instead of a poorly-identified nonlinear fit. L = 0.716 puts the
curvature peak near 2.5y for maturities in years.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

LAMBDA = 0.716


def ns_loadings(maturities=None, lam: float = LAMBDA) -> np.ndarray:
    """Fixed 3-column design matrix [level | slope | curvature]. Shape (N, 3)."""
    tau = np.asarray(maturities if maturities is not None else config.MATURITIES,
                     dtype=float)
    decay = (1 - np.exp(-lam * tau)) / (lam * tau)
    return np.column_stack([np.ones_like(tau), decay, decay - np.exp(-lam * tau)])


@dataclass
class NSResult:
    betas: pd.DataFrame           # T x 3
    rmse_bps: float
    lam: float


def fit_nelson_siegel(levels: pd.DataFrame, lam: float = LAMBDA) -> NSResult:
    """Fit betas for every day by OLS. Lambda fixed, so the design matrix is the
    same every day and lstsq solves all days at once."""
    H = ns_loadings(config.MATURITIES, lam)          # (N, 3)
    Y = levels[config.COLUMNS].values.T              # (N, T)
    betas, *_ = np.linalg.lstsq(H, Y, rcond=None)    # (3, T)
    resid_bps = (Y - H @ betas) * 100
    df = pd.DataFrame(betas.T, index=levels.index,
                      columns=["ns_level", "ns_slope", "ns_curvature"])
    return NSResult(df, float(np.sqrt((resid_bps ** 2).mean())), lam)


def factor_correlation(ns: NSResult, pc_scores: pd.DataFrame) -> pd.DataFrame:
    """Correlate daily CHANGES in the NS betas with the PC scores. PCA runs on
    changes so its scores already are changes; the betas are levels, so they get
    differenced first. Signs are convention-dependent, so read the magnitudes."""
    d = ns.betas.diff().dropna()
    common = d.index.intersection(pc_scores.index)
    d, pcs = d.loc[common], pc_scores.loc[common, ["PC1", "PC2", "PC3"]]
    corr = np.array([[np.corrcoef(d[a], pcs[b])[0, 1] for b in pcs.columns]
                     for a in d.columns])
    return pd.DataFrame(corr, index=["Δ NS level", "Δ NS slope", "Δ NS curv"],
                        columns=["PC1", "PC2", "PC3"])


def diagonal_strength(corr: pd.DataFrame) -> dict:
    """Diagonal |corr| (should be high) vs max off-diagonal (should be low)."""
    C = corr.values
    diag = np.abs(np.diag(C))
    off = np.abs(C - np.diag(np.diag(C)))
    return {"min_diagonal_abs_corr": float(diag.min()),
            "mean_diagonal_abs_corr": float(diag.mean()),
            "max_offdiagonal_abs_corr": float(off.max())}
