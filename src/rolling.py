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

    # Match every one of the first k new eigenvectors to the previous window's,
    # by largest |cosine|. Solved as an assignment problem so it works for any
    # k -- the earlier version only ever checked the PC2/PC3 pair, so calling
    # this with k=2 silently disabled swap detection entirely.
    M = np.abs(V_prev[:, :k].T @ V[:, :k])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-M)
    except ImportError:                        # greedy fallback, no scipy
        cols, taken = [], set()
        for i in range(k):
            order = np.argsort(-M[i])
            pick = next(j for j in order if j not in taken)
            taken.add(pick)
            cols.append(pick)
        rows, cols = np.arange(k), np.array(cols)

    swapped = bool((cols != np.arange(k)).any())
    idx = np.arange(V.shape[1])
    idx[:k] = cols
    V, lam = V[:, idx], lam[idx]

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
    if len(changes) < window:
        raise ValueError(f"Need at least {window} observations for a "
                         f"{window}-day window, got {len(changes)}.")
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