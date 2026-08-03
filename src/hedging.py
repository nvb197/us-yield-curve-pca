"""Hedging demo (Master Doc B1): duration hedge vs PC hedge.

Classic story: a "2s10s" duration-neutral curve trade is immune to PARALLEL
shifts by construction (that IS duration hedging), but has full exposure to
the SLOPE factor -- exactly what makes duration hedging insufficient in a
bear-flattening episode like 2022-23. A genuine PC-hedge (two instruments,
solved so total exposure is orthogonal to BOTH v1 and v2 -- doc E.7 KRD
formula) neutralizes level AND slope risk, leaving only curvature (PC3) and
higher-order residual as unhedged risk.

Durations here are STYLIZED approximate modified durations for a par bond at
each maturity (doc E.4/E.7) -- illustrative rule-of-thumb values, not fitted
from real bond prices. Swap in real durations (e.g. from a pricing library)
for production use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

APPROX_DURATION = {"m3": 0.25, "m6": 0.5, "y1": 1.0, "y2": 1.9, "y3": 2.8,
                   "y5": 4.5, "y7": 6.0, "y10": 8.5}


@dataclass
class Position:
    """Dollar-duration exposure vector (doc E.7): w[i] = $ P&L per 1bp move
    at tenor i for a LONG holding (loses money if y_i rises). A short
    position at tenor i contributes a negative w[i]."""
    w: np.ndarray             # length len(config.COLUMNS), $ per bp
    label: str

    def pnl(self, delta_y: pd.Series) -> float:
        """P&L in $ for a curve move delta_y (percentage points, project-
        wide convention). Delta_P = -sum_i w_i * delta_y_i(bps)."""
        dy = delta_y.reindex(config.COLUMNS)
        if dy.isna().any():
            missing = [c for c, v in zip(config.COLUMNS, dy.isna()) if v]
            raise ValueError(f"delta_y is missing tenors {missing}; a partial "
                             f"curve would silently produce a NaN P&L.")
        return float(-(self.w * dy.values * 100).sum())

    def exposure(self, V_cov: np.ndarray, k: int = 3) -> np.ndarray:
        """w . v_j for j=1..k -- residual exposure to each PC direction.
        Zero means fully neutral to that factor.

        Units: dollars per one unit of factor amplitude, where amplitude is
        measured in percentage points (the units v_j lives in). That is 100x
        the dollars-per-basis-point figure, so divide by 100 to compare these
        against DV01. Only ratios and zeros matter for hedging, which is what
        this is used for -- but do not read the raw numbers as DV01."""
        return np.array([self.w @ V_cov[:, j] for j in range(k)])


def long_position(tenor: str, notional: float) -> Position:
    i = config.COLUMNS.index(tenor)
    w = np.zeros(len(config.COLUMNS))
    w[i] = notional * APPROX_DURATION[tenor] * 0.0001
    return Position(w, f"Long ${notional/1e6:.0f}mm {tenor.upper()}")


def duration_hedge(base: Position, hedge_tenor: str) -> Position:
    """Offset at `hedge_tenor` sized to zero NET dollar-duration
    (sum_i w_i = 0) -- neutral to perfectly parallel shifts only."""
    j = config.COLUMNS.index(hedge_tenor)
    w = base.w.copy()
    w[j] += -base.w.sum()
    return Position(w, base.label + f" + duration-hedge @ {hedge_tenor.upper()}")


def pc_hedge(base: Position, V_cov: np.ndarray,
            hedge_tenors: tuple[str, str]) -> Position:
    """Offset at TWO tenors, notionals solved so total exposure satisfies
    w.v1 = 0 AND w.v2 = 0 simultaneously -- neutral to PC1 (level) AND PC2
    (slope), not just parallel shocks. A 2x2 linear system (doc E.7)."""
    j1, j2 = (config.COLUMNS.index(t) for t in hedge_tenors)
    v1, v2 = V_cov[:, 0], V_cov[:, 1]
    A = np.array([[v1[j1], v1[j2]], [v2[j1], v2[j2]]])
    b = -np.array([base.w @ v1, base.w @ v2])
    # Two instruments whose factor loadings are near-collinear cannot span the
    # PC1/PC2 plane: the system is singular and any "solution" would demand
    # enormous offsetting notionals. Fail loudly with the diagnostic rather
    # than letting LinAlgError escape or returning a meaningless hedge.
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e6:
        raise ValueError(
            f"Hedge instruments {hedge_tenors} have near-collinear factor "
            f"loadings (condition number {cond:.3g}). Pick tenors further "
            f"apart on the curve.")
    x1, x2 = np.linalg.solve(A, b)
    w = base.w.copy()
    w[j1] += x1
    w[j2] += x2
    label = base.label + f" + PC-hedge @ {hedge_tenors[0].upper()}/{hedge_tenors[1].upper()}"
    return Position(w, label)


def pnl_table(positions: list[Position],
             episode_deltas: dict[str, pd.Series]) -> pd.DataFrame:
    """P&L (rounded, $) of each position across each episode's realized
    curve move ΔY (doc 5.4) -- reuse events.run_event_study()'s `delta`."""
    rows = {}
    for name, dY in episode_deltas.items():
        rows[name] = {p.label: round(p.pnl(dY)) for p in positions}
    return pd.DataFrame(rows).T