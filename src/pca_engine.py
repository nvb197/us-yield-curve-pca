"""PCA core, written by hand around numpy.linalg.eigh (no black box).

Math (doc section 3): for centered (optionally standardized) X in R^{T x N},
Sigma = X'X/(T-1); solve Sigma v = lambda v. Sigma is symmetric PSD, so the
spectral theorem gives an orthonormal eigenbasis with lambda >= 0. `eigh`
returns eigenvalues ASCENDING -> we reverse (classic bug #1).

mode="corr" (z-score) == PCA on the correlation matrix (structure view).
mode="cov"  (center only) == PCA on the covariance matrix (units stay in
percentage points -> use THIS basis for event-study decompositions and risk).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def first_diff(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily changes, in percentage points (x100 = bps)."""
    return levels.diff().dropna()


def adf_table(levels: pd.DataFrame) -> pd.DataFrame:
    """ADF p-values on levels vs first differences (doc 3.1).
    Expected: levels fail to reject unit root; diffs reject strongly."""
    from statsmodels.tsa.stattools import adfuller

    rows = {}
    d = first_diff(levels)
    for col in levels.columns:
        rows[col] = {
            "p_level": adfuller(levels[col].values, autolag="AIC")[1],
            "p_diff": adfuller(d[col].values, autolag="AIC")[1],
        }
    return pd.DataFrame(rows).T.round(4)


@dataclass
class PCAResult:
    eigenvalues: np.ndarray          # descending
    eigenvectors: np.ndarray         # columns, sign-conventioned
    evr: np.ndarray                  # explained variance ratio
    scores: pd.DataFrame             # T x N, columns PC1..PCN
    mean: np.ndarray
    std: np.ndarray | None           # None in cov mode
    columns: list = field(default_factory=list)
    mode: str = "corr"

    @property
    def effective_rank(self) -> float:
        """exp(spectral entropy) — the market's 'effective dimension'
        (Roy & Vetterli 2007)."""
        p = self.eigenvalues / self.eigenvalues.sum()
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))


def _sign_convention(V: np.ndarray) -> np.ndarray:
    """Eigenvectors are defined up to +-. Fix: PC1 all-positive (level);
    PC2, PC3 have the LONGEST-tenor loading positive (columns are ordered
    short -> long, so that's the last row)."""
    V = V.copy()
    if V[:, 0].sum() < 0:
        V[:, 0] *= -1
    for j in range(1, V.shape[1]):
        if V[-1, j] < 0:
            V[:, j] *= -1
    return V


def fit_pca(changes: pd.DataFrame, mode: str = "corr") -> PCAResult:
    X = changes.values.astype(float)
    mean = X.mean(axis=0)
    Xc = X - mean
    std = None
    if mode == "corr":
        std = Xc.std(axis=0, ddof=1)
        # A tenor pinned at zero (ZIRP, a stale feed) has zero variance, and
        # dividing by it turns the column into NaN, which makes eigh fail to
        # converge. Such a column carries no information, so leave it at zero
        # rather than propagating NaN through the whole decomposition.
        dead = std < 1e-12
        if dead.any():
            import warnings
            warnings.warn(f"{dead.sum()} tenor(s) have zero variance and were "
                          f"left unscaled: {[c for c, d in zip(changes.columns, dead) if d]}",
                          RuntimeWarning)
            std = np.where(dead, 1.0, std)
        Xc = Xc / std
    elif mode != "cov":
        raise ValueError("mode must be 'corr' or 'cov'")

    sigma = Xc.T @ Xc / (len(Xc) - 1)
    lam, V = np.linalg.eigh(sigma)          # ASCENDING ->
    order = np.argsort(lam)[::-1]           # reverse to descending
    lam, V = lam[order], V[:, order]
    lam = np.clip(lam, 0, None)             # kill -1e-17 round-off
    V = _sign_convention(V)

    scores = pd.DataFrame(Xc @ V, index=changes.index,
                          columns=[f"PC{k+1}" for k in range(V.shape[1])])
    return PCAResult(lam, V, lam / lam.sum(), scores, mean, std,
                     list(changes.columns), mode)


def reconstruct(res: PCAResult, k: int) -> pd.DataFrame:
    """Rebuild the changes from the first k components (doc 3.5)."""
    Xk = res.scores.values[:, :k] @ res.eigenvectors[:, :k].T
    if res.std is not None:
        Xk = Xk * res.std
    return pd.DataFrame(Xk + res.mean, index=res.scores.index,
                        columns=res.columns)


def rmse_table(changes: pd.DataFrame, res: PCAResult,
               ks=(1, 2, 3)) -> pd.DataFrame:
    """RMSE per tenor in BPS for k = 1..3 — the concrete meaning of
    'three numbers compress eight dimensions'."""
    out = {}
    for k in ks:
        err = (changes - reconstruct(res, k)) * 100  # pp -> bps
        out[f"k={k}"] = np.sqrt((err ** 2).mean())
    return pd.DataFrame(out).round(2)


def sklearn_crosscheck(changes: pd.DataFrame, res: PCAResult) -> float:
    """Max abs deviation vs sklearn (expect ~1e-12). 'I don't trust the
    black box, I verify it.'"""
    from sklearn.decomposition import PCA as SkPCA

    X = changes.values - res.mean
    if res.std is not None:
        X = X / res.std
    sk = SkPCA(n_components=len(res.columns)).fit(X)
    dev = 0.0
    for j in range(len(res.columns)):
        v, w = res.eigenvectors[:, j], sk.components_[j]
        dev = max(dev, float(np.abs(v - np.sign(v @ w) * w).max()))
    dev = max(dev, float(np.abs(sk.explained_variance_ - res.eigenvalues).max()))
    return dev