"""Statistical appendix (project appendix): permutation test answers
"is the 3-factor structure real or could it arise from independent noise?",
block bootstrap answers "how precisely do I know EVR1, given that?".

Both live here because they share the same resampling skeleton (doc G.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pca_engine import fit_pca


@dataclass
class PermutationResult:
    observed_evr1: float
    null_evr1: np.ndarray          # EVR1 under H0: tenors independent
    p_value: float                 # P(null >= observed) -- one-sided

    @property
    def null_mean(self) -> float:
        return float(self.null_evr1.mean())


def permutation_test(changes: pd.DataFrame, n_perm: int = 1000,
                     seed: int = 0, mode: str = "corr") -> PermutationResult:
    """Independently shuffle the TIME ORDER of each column separately. This
    destroys cross-tenor correlation (what PCA needs to find structure)
    while preserving each column's own marginal distribution (mean, vol,
    fat tails stay identical) -- so the null is a fair "what if the 8
    tenors were unrelated" benchmark, not a strawman.
    """
    rng = np.random.default_rng(seed)
    X = changes.values
    T, N = X.shape
    observed = fit_pca(changes, mode=mode).evr[0]

    null = np.empty(n_perm)
    for b in range(n_perm):
        Xp = np.empty_like(X)
        for j in range(N):
            Xp[:, j] = X[rng.permutation(T), j]
        res = fit_pca(pd.DataFrame(Xp, columns=changes.columns), mode=mode)
        null[b] = res.evr[0]

    p = float((null >= observed).mean())
    return PermutationResult(observed, null, max(p, 1.0 / (n_perm + 1)))


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lo: float
    ci_hi: float
    block_length: int
    samples: np.ndarray


def block_bootstrap_evr1(changes: pd.DataFrame, n_boot: int = 1000,
                         block_length: int | None = None, seed: int = 0,
                         mode: str = "corr", ci: float = 0.95) -> BootstrapResult:
    """Circular block bootstrap for a CI on EVR1. Block length defaults to
    round(T^(1/3)) (doc B1's justification), snapped to a trading month.
    Circular wrap-around avoids under-sampling the series' tail.
    """
    T = len(changes)
    if block_length is None:
        # T^(1/3) is the standard rule of thumb; snap it to whole trading
        # months for interpretability, but never below one month. A block of
        # 1 would silently degrade this into an iid bootstrap and destroy the
        # autocorrelation structure the block scheme exists to preserve.
        block_length = max(21, round(T ** (1 / 3) / 21) * 21)
    if block_length >= T:
        raise ValueError(f"block_length {block_length} must be shorter than "
                         f"the sample ({T} observations).")
    rng = np.random.default_rng(seed)
    X = changes.values
    n_blocks = int(np.ceil(T / block_length))
    point = fit_pca(changes, mode=mode).evr[0]

    samples = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) % T for s in starts])[:T]
        Xb = X[idx]
        samples[b] = fit_pca(pd.DataFrame(Xb, columns=changes.columns),
                             mode=mode).evr[0]

    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return BootstrapResult(point, float(lo), float(hi), block_length, samples)