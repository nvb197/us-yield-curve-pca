"""Robustness checks (Master Doc §2.2, Q2, Q12).

A result that only holds for one arbitrary choice of tenor set / window
length / sample period is not a finding, it's an artefact. Every claim in
the README should survive these four checks — and where it doesn't, that
becomes a stated limitation rather than a hidden one.

Four axes:
1. tenor_sensitivity   — does the 3-factor structure depend on WHICH tenors?
2. window_sensitivity  — does the rolling story depend on the 252-day choice?
3. subperiod_stability — do loadings drift across decades?
4. mode_sensitivity    — correlation vs covariance basis (doc §3.2 / C.7)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .pca_engine import first_diff, fit_pca
from .rolling import rolling_pca


def tenor_sensitivity(levels: pd.DataFrame,
                      subsets: dict[str, list[str]] | None = None,
                      mode: str = "corr") -> pd.DataFrame:
    """Re-run PCA on different tenor subsets. The level/slope/curvature
    structure should survive dropping the ends or thinning the middle — if
    EVR shares swing wildly, the 'three factors' claim is tenor-specific."""
    cols = config.COLUMNS
    if subsets is None:
        subsets = {
            "core (all 8)": cols,
            "drop short end (1Y+)": [c for c in cols if c not in ("m3", "m6")],
            "drop long end (<=5Y)": [c for c in cols
                                     if c in ("m3", "m6", "y1", "y2", "y3", "y5")],
            "sparse (3M,2Y,10Y)": ["m3", "y2", "y10"],
            "belly only (1Y-7Y)": ["y1", "y2", "y3", "y5", "y7"],
        }
    rows = []
    for name, sub in subsets.items():
        if not set(sub).issubset(levels.columns):
            continue
        res = fit_pca(first_diff(levels[sub]), mode=mode)
        rows.append({
            "subset": name, "n_tenors": len(sub),
            "EVR1": res.evr[0],
            "EVR2": res.evr[1] if len(res.evr) > 1 else np.nan,
            "EVR3": res.evr[2] if len(res.evr) > 2 else np.nan,
            "cum_top3": res.evr[:3].sum(),
            "eff_rank": res.effective_rank,
            # normalised so tenor sets of different size are comparable:
            # raw effective rank is bounded by N, so it shrinks mechanically
            # when you drop tenors.
            "eff_rank_pct_of_N": res.effective_rank / len(sub),
        })
    return pd.DataFrame(rows).set_index("subset").round(4)


def window_sensitivity(changes: pd.DataFrame,
                       windows=(126, 252, 504),
                       step: int = 21, mode: str = "corr") -> pd.DataFrame:
    """Does the rolling narrative depend on the window length? Baseline
    level should shift a little; the crisis SPIKES should stay put."""
    rows = []
    for w in windows:
        if len(changes) < w + step:
            continue
        summary, _ = rolling_pca(changes, window=w, step=step, mode=mode)
        peak_date = summary["evr1"].idxmax()
        rows.append({
            "window": w, "n_windows": len(summary),
            "EVR1_median": summary["evr1"].median(),
            "EVR1_p95": summary["evr1"].quantile(0.95),
            "EVR1_max": summary["evr1"].max(),
            "peak_date": str(peak_date.date()),
            "erank_median": summary["erank"].median(),
            "erank_min": summary["erank"].min(),
        })
    return pd.DataFrame(rows).set_index("window").round(4)


def subperiod_stability(levels: pd.DataFrame, n_periods: int = 3,
                        mode: str = "corr") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sample into equal periods, fit PCA on each, and report both
    EVR and the ANGLE between each period's PC1 and the full-sample PC1.

    The angle is the honest test: |cos| near 1 (angle near 0 deg) means the
    factor points the same way across decades; a large angle means the
    'level factor' is not the same object in 2003 and 2023."""
    changes = first_diff(levels)
    full = fit_pca(changes, mode=mode)
    chunks = np.array_split(np.arange(len(changes)), n_periods)

    evr_rows, ang_rows = [], []
    for i, idx in enumerate(chunks, start=1):
        sub = changes.iloc[idx]
        res = fit_pca(sub, mode=mode)
        label = f"P{i}: {sub.index[0].date()}→{sub.index[-1].date()}"
        evr_rows.append({"period": label, "n_days": len(sub),
                         "EVR1": res.evr[0], "EVR2": res.evr[1],
                         "EVR3": res.evr[2], "cum_top3": res.evr[:3].sum()})
        angles = {}
        for k in range(3):
            cos = abs(float(res.eigenvectors[:, k] @ full.eigenvectors[:, k]))
            angles[f"angle_PC{k+1}_deg"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ang_rows.append({"period": label, **angles})

    return (pd.DataFrame(evr_rows).set_index("period").round(4),
            pd.DataFrame(ang_rows).set_index("period").round(2))


def mode_sensitivity(changes: pd.DataFrame) -> pd.DataFrame:
    """Correlation vs covariance basis, side by side (doc §3.2 / C.7):
    two genuinely different optimisation problems, not preprocessing."""
    rows = []
    for mode in ("corr", "cov"):
        res = fit_pca(changes, mode=mode)
        row = {"mode": mode, "EVR1": res.evr[0], "EVR2": res.evr[1],
               "EVR3": res.evr[2], "cum_top3": res.evr[:3].sum()}
        for i, col in enumerate(changes.columns):
            row[f"PC1_{col}"] = res.eigenvectors[i, 0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("mode").round(4)


def run_all_sensitivity(levels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    changes = first_diff(levels)
    evr, ang = subperiod_stability(levels)
    return {
        "tenor": tenor_sensitivity(levels),
        "window": window_sensitivity(changes),
        "subperiod_evr": evr,
        "subperiod_angles": ang,
        "mode": mode_sensitivity(changes),
    }