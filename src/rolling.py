"""Rolling-window PCA with eigenvector alignment (doc section 5).

The theory problem made practical: eigenvectors are defined only up to sign,
and up to rotation inside an eigenspace when eigenvalues are (near-)equal.
Across rolling windows this shows up as (a) random sign flips, (b) PC2/PC3
swapping order when lambda2 ~ lambda3. Unhandled, loading time-series jump
around and the plots are garbage.

Pragmatic alignment (doc 5.2): lambda1 is so dominant here that PC1 never
swaps; only the PC2<->PC3 pair can. So per window: (i) swap-check that pair
against the previous ALIGNED window, (ii) sign-fix every kept column by dot
product with the previous one. The general problem (full matching /
Hungarian) is discussed in the README; with N=8 and a well-separated
spectrum this 15-line version is equivalent.

Standardization is computed INSIDE each window (mu, sigma of that window
only). Standardizing on the full sample first would leak future information
into past windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .pca_engine import fit_pca


def align_to_previous(V: np.ndarray, lam: np.ndarray,
                      V_prev: np.ndarray, k: int = 3):
    """Return (V, lam, swapped) aligned with the previous window."""
    V, lam = V.copy(), lam.copy()
    swapped = False
    if k >= 3:
        keep = abs(float(V_prev[:, 1] @ V[:, 1]))
        cross = abs(float(V_prev[:, 1] @ V[:, 2]))
        if cross > keep:                       # PC2/PC3 traded places
            V[:, [1, 2]] = V[:, [2, 1]]
            lam[[1, 2]] = lam[[2, 1]]
            swapped = True
    for j in range(k):
        if float(V_prev[:, j] @ V[:, j]) < 0:  # sign flip
            V[:, j] *= -1
    return V, lam, swapped


def rolling_pca(changes: pd.DataFrame,
                window: int = config.ROLL_WINDOW,
                step: int = config.ROLL_STEP,
                mode: str = "corr",
                k: int = config.N_FACTORS):
    """Returns (summary DataFrame indexed by window-end date,
    loadings dict PCj -> DataFrame[date x tenor])."""
    rows, loads = [], {j: [] for j in range(k)}
    V_prev = None
    n_swaps = 0
    for start in range(0, len(changes) - window + 1, step):
        win = changes.iloc[start:start + window]
        res = fit_pca(win, mode=mode)
        V, lam = res.eigenvectors, res.eigenvalues
        if V_prev is not None:
            V, lam, sw = align_to_previous(V, lam, V_prev, k)
            n_swaps += sw
        V_prev = V
        p = lam / lam.sum()
        end = win.index[-1]
        rows.append({"date": end,
                     **{f"evr{j+1}": p[j] for j in range(k)},
                     "erank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))})
        for j in range(k):
            loads[j].append(pd.Series(V[:, j], index=changes.columns, name=end))
    summary = pd.DataFrame(rows).set_index("date")
    loadings = {f"PC{j+1}": pd.DataFrame(loads[j]) for j in range(k)}
    summary.attrs["n_swaps"] = n_swaps
    return summary, loadings
