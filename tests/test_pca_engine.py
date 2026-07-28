"""Tests per doc 4.2 (+ alignment and event-study invariants).
Run: pytest -q  (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.events import decompose_episode
from src.etl import clean_yields
from src.pca_engine import first_diff, fit_pca, reconstruct
from src.rolling import align_to_previous, rolling_pca
from src.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    levels, _ = make_synthetic()
    changes = first_diff(levels)
    return levels, changes, fit_pca(changes, mode="corr")


def test_eigenvectors_orthonormal(data):
    _, _, res = data
    V = res.eigenvectors
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)


def test_eigenvalues_sorted_positive_evr_sums_to_one(data):
    _, _, res = data
    lam = res.eigenvalues
    assert np.all(lam[:-1] >= lam[1:]) and np.all(lam >= 0)
    assert np.isclose(res.evr.sum(), 1.0)


def test_reconstruction_exact_with_all_components(data):
    _, changes, res = data
    full = reconstruct(res, k=len(config.COLUMNS))
    assert np.allclose(full.values, changes.values, atol=1e-10)


def test_recovers_three_factor_structure(data):
    """Synthetic data has 3 factors + small noise: EVR of top-3 must
    dominate, lambda4 must be noise-sized. (No eigenvector matching —
    sign/rotation issues make that a bad unit test; doc 4.2.)"""
    _, _, res = data
    assert res.evr[:3].sum() > 0.95
    assert res.evr[3] < 0.02


def test_sign_convention(data):
    _, _, res = data
    V = res.eigenvectors
    assert V[:, 0].sum() > 0          # PC1 all-level positive
    assert V[-1, 1] > 0 and V[-1, 2] > 0  # longest tenor positive on PC2/PC3


def test_clean_yields_drops_asymmetric_gaps():
    idx = pd.bdate_range("2020-01-01", periods=10)
    raw = pd.DataFrame(1.0, index=idx, columns=config.COLUMNS)
    raw.iloc[3, 0] = np.nan            # isolated 1-day gap -> ffilled
    raw.iloc[5:8, 2] = np.nan          # 3-day gap -> days dropped
    clean, log = clean_yields(raw, ffill_limit=1)
    assert log["n_ffilled"] == 2       # day3(col0) + day5 (first of 3-day gap)
    assert not clean.isna().any().any()
    assert len(clean) == 10 - 2        # days 6,7 (unfillable) are dropped


def test_alignment_fixes_sign_and_swap():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    V_prev = Q
    V_new = Q.copy()
    V_new[:, [1, 2]] = V_new[:, [2, 1]]     # swap PC2/PC3
    V_new[:, 0] *= -1                        # flip PC1 sign
    lam = np.arange(8, 0, -1).astype(float)
    V_al, lam_al, swapped = align_to_previous(V_new, lam, V_prev, k=3)
    assert swapped
    for j in range(3):
        assert np.allclose(V_al[:, j], V_prev[:, j])
    assert lam_al[1] == lam[2] and lam_al[2] == lam[1]


def test_rolling_runs_and_erank_bounds():
    levels, _ = make_synthetic()
    changes = first_diff(levels).iloc[:1500]
    summary, loadings = rolling_pca(changes, window=252, step=63)
    assert (summary["erank"] >= 1).all()
    assert (summary["erank"] <= len(config.COLUMNS)).all()
    assert not loadings["PC1"].isna().any().any()


def test_event_decomposition_residual_zero_in_span():
    """If DeltaY lies exactly in the span of the top-3 eigenvectors, the
    residual must vanish — the decomposition is a true projection."""
    levels, _ = make_synthetic()
    changes = first_diff(levels)
    res = fit_pca(changes, mode="cov")
    V = res.eigenvectors
    target = levels.iloc[100] + pd.Series(
        0.5 * V[:, 0] - 0.2 * V[:, 1], index=config.COLUMNS)
    lv = levels.copy()
    lv.iloc[200] = target
    d = decompose_episode(lv, V, str(lv.index[100].date()),
                          str(lv.index[200].date()))
    assert d["residual_share"] < 1e-10


# --- hedging.py (Master Doc B1) ------------------------------------------

def test_duration_hedge_neutralizes_parallel_not_slope():
    from src.hedging import long_position, duration_hedge

    base = long_position("y10", 100e6)
    dh = duration_hedge(base, "y2")
    assert np.isclose(dh.w.sum(), 0.0)             # duration-neutral

    parallel = pd.Series(0.10, index=config.COLUMNS)
    assert np.isclose(dh.pnl(parallel), 0.0, atol=1e-6)   # protected

    slope = pd.Series(0.0, index=config.COLUMNS)
    slope["y2"], slope["y10"] = 0.10, -0.10
    assert abs(dh.pnl(slope)) > 1e6                 # NOT protected — the point


def test_pc_hedge_zeroes_pc1_pc2_exposure_exactly():
    from src.hedging import long_position, pc_hedge

    levels, _ = make_synthetic()
    res = fit_pca(first_diff(levels), mode="cov")
    base = long_position("y10", 100e6)
    hedged = pc_hedge(base, res.eigenvectors, ("y2", "y5"))
    exposure = hedged.exposure(res.eigenvectors, k=3)
    assert np.allclose(exposure[:2], 0.0, atol=1e-6)   # PC1, PC2 neutral
    assert abs(exposure[2]) > 1e-3                      # PC3 residual remains


# --- resampling.py (Master Doc B2) ----------------------------------------

def test_permutation_null_centers_near_one_over_n():
    from src.resampling import permutation_test

    levels, _ = make_synthetic()
    changes = first_diff(levels).iloc[:1000]
    result = permutation_test(changes, n_perm=100, seed=1)
    assert abs(result.null_mean - 1 / len(config.COLUMNS)) < 0.03
    assert result.observed_evr1 > result.null_evr1.max()   # real structure


def test_bootstrap_ci_brackets_point_estimate():
    from src.resampling import block_bootstrap_evr1

    levels, _ = make_synthetic()
    changes = first_diff(levels)   # full sample: T^(1/3)~19 -> rounds to 21
    result = block_bootstrap_evr1(changes, n_boot=100, seed=1)
    assert result.ci_lo < result.point_estimate < result.ci_hi
    assert result.block_length == 21


# --- sensitivity.py (robustness) ------------------------------------------

def test_three_factor_structure_survives_tenor_subsets():
    """The headline claim must not depend on which tenors we picked."""
    from src.sensitivity import tenor_sensitivity

    levels, _ = make_synthetic()
    tbl = tenor_sensitivity(levels)
    assert (tbl["cum_top3"] > 0.90).all()      # holds for every subset
    assert len(tbl) >= 4


def test_subperiod_factors_point_the_same_way():
    """Angle between each subperiod's PC1 and the full-sample PC1 must be
    small — otherwise 'the level factor' isn't the same object over time."""
    from src.sensitivity import subperiod_stability

    levels, _ = make_synthetic()
    evr, angles = subperiod_stability(levels, n_periods=3)
    assert len(evr) == 3
    assert (angles["angle_PC1_deg"] < 20).all()
    assert (angles >= 0).all().all()           # angles are well-defined


def test_window_sensitivity_preserves_crisis_spikes():
    from src.sensitivity import window_sensitivity

    levels, _ = make_synthetic()
    tbl = window_sensitivity(first_diff(levels), windows=(126, 252))
    assert len(tbl) == 2
    # synthetic crises are engineered at 4x level vol -> max must exceed median
    assert (tbl["EVR1_max"] > tbl["EVR1_median"] + 0.05).all()


# --- queries.py (SQL layer) ------------------------------------------------

def test_sql_queries_run_against_a_real_db(tmp_path):
    """Build a throwaway SQLite DB through the real ETL path, then check
    every analytical query executes and returns sane shapes."""
    from src import etl, queries

    levels, usrec = make_synthetic()
    clean, log = etl.clean_yields(levels, ffill_limit=1)
    db = tmp_path / "test.db"
    etl.write_db(levels, clean, usrec, log, db_path=db)

    yearly = queries.yearly_summary(db)
    assert len(yearly) > 20                       # ~26 years
    assert {"year", "avg_10y", "pct_inverted"}.issubset(yearly.columns)
    assert yearly["trading_days"].sum() == len(clean)

    regimes = queries.regime_day_counts(db)
    assert regimes["n_days"].sum() == len(clean) - 1   # first day has no LAG
    assert set(regimes["regime"]).issubset(
        {"bear_flattening", "bull_steepening", "bear_steepening",
         "bull_flattening", "flat_day"})

    streaks = queries.longest_inversion_streaks(db)
    assert list(streaks.columns) == ["start_date", "end_date",
                                     "trading_days_inverted"]
    assert queries.etl_audit_log(db).shape[0] == 1


# --- nelson_siegel.py ----------------------------------------------------

def test_ns_loadings_have_level_slope_curvature_shape():
    from src.nelson_siegel import ns_loadings

    H = ns_loadings()
    assert np.allclose(H[:, 0], 1.0)                # level: flat
    assert H[0, 1] > H[-1, 1]                        # slope: decays
    assert H[0, 1] > 0.8 and H[-1, 1] < 0.3
    assert H[:, 2].argmax() not in (0, len(H) - 1)   # curvature: humped


def test_ns_and_pca_factors_agree():
    """Two unrelated methods should recover the same three factors."""
    from src.nelson_siegel import (fit_nelson_siegel, factor_correlation,
                                   diagonal_strength)

    levels, _ = make_synthetic()
    ns = fit_nelson_siegel(levels)
    scores = fit_pca(first_diff(levels), mode="corr").scores
    d = diagonal_strength(factor_correlation(ns, scores))
    assert d["mean_diagonal_abs_corr"] > 0.80
    assert d["max_offdiagonal_abs_corr"] < d["min_diagonal_abs_corr"]
