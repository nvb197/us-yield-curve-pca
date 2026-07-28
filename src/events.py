"""Event study: decompose each macro episode's total curve move into factor
contributions (doc section 5.4).

Method: for an episode [t0, t1], DeltaY = y(t1) - y(t0) in R^8 (percentage
points). Project onto the FULL-SAMPLE covariance-mode eigenvectors:
contribution of factor k is (DeltaY . v_k) v_k. We use the COVARIANCE basis
here on purpose: it is orthonormal in raw yield space, so contributions are
in interpretable units (pp/bps). The correlation basis lives in z-score
space and would mix units.

Anti-cherry-picking rules (doc 5.4 / pitfall #11):
1. t0/t1 are anchored EX ANTE to documented external events (see
   config.EPISODES, field `anchors`) — never chosen from the chart.
2. Residual share ||DeltaY - sum_k c_k|| / ||DeltaY|| is reported for every
   episode. A large residual is discussed, not hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _asof(levels: pd.DataFrame, date: str | None) -> pd.Series:
    """Curve on `date`, or the last trading day before it. None = latest."""
    if date is None:
        return levels.iloc[-1]
    ts = pd.Timestamp(date)
    sub = levels.loc[:ts]
    if sub.empty:
        raise KeyError(f"No data on or before {date}")
    return sub.iloc[-1]


def decompose_episode(levels: pd.DataFrame, V_cov: np.ndarray,
                      t0: str, t1: str | None, k: int = 3) -> dict:
    """Returns dict with DeltaY, per-factor contribution vectors, residual,
    residual_share, and scalar factor amplitudes (DeltaY . v_j)."""
    y0, y1 = _asof(levels, t0), _asof(levels, t1)
    dY = (y1 - y0).values.astype(float)
    contribs, amps = {}, {}
    total = np.zeros_like(dY)
    for j in range(k):
        v = V_cov[:, j]
        a = float(dY @ v)
        c = a * v
        contribs[f"PC{j+1}"] = c
        amps[f"PC{j+1}"] = a
        total += c
    resid = dY - total
    return dict(
        t0=str(y0.name.date()), t1=str(y1.name.date()),
        delta=pd.Series(dY, index=levels.columns),
        contribs={n: pd.Series(c, index=levels.columns)
                  for n, c in contribs.items()},
        residual=pd.Series(resid, index=levels.columns),
        residual_share=float(np.linalg.norm(resid) / np.linalg.norm(dY)),
        amplitudes=amps,
    )


def run_event_study(levels: pd.DataFrame, V_cov: np.ndarray,
                    episodes: dict | None = None, k: int = 3) -> dict:
    episodes = episodes or config.EPISODES
    out = {}
    for key, ep in episodes.items():
        try:
            d = decompose_episode(levels, V_cov, ep["t0"], ep["t1"], k)
        except KeyError:
            continue  # episode outside available data range
        d["label"], d["expected"] = ep["label"], ep["expected"]
        d["anchors"] = ep["anchors"]
        out[key] = d
    return out


def summary_table(study: dict) -> pd.DataFrame:
    rows = []
    for key, d in study.items():
        rows.append({
            "episode": d["label"], "t0": d["t0"], "t1": d["t1"],
            **{f"{n}_bps": round(a * 100, 1) for n, a in d["amplitudes"].items()},
            "residual_share": round(d["residual_share"], 3),
        })
    return pd.DataFrame(rows).set_index("episode")
