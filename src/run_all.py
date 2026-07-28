"""Single entrypoint: `python -m src.run_all` reproduces every figure/table.

Real data:   export FRED_API_KEY, then `python -m src.run_all`
             (fetches on first run, then reads data/yield_data.db)
Offline dev: `python -m src.run_all --synthetic` runs the identical pipeline
             on generated 3-factor data (plumbing check only, NOT results)

Stages can be run selectively: --stages core,events,rolling,hedging,stats,sensitivity,sql
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, plots
from .events import run_event_study, summary_table
from .pca_engine import (adf_table, first_diff, fit_pca, rmse_table,
                         sklearn_crosscheck)
from .rolling import rolling_pca

ALL_STAGES = ("core", "events", "rolling", "hedging", "stats",
              "nelson_siegel", "sensitivity", "sql")


def _h(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="generated 3-factor data; no FRED key needed")
    ap.add_argument("--stages", default="all",
                    help=f"comma-separated subset of {ALL_STAGES}, or 'all'")
    ap.add_argument("--n-resamples", type=int, default=500,
                    help="permutation/bootstrap draws (default 500)")
    args = ap.parse_args()

    stages = ALL_STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(","))
    for s in stages:
        if s not in ALL_STAGES:
            raise SystemExit(f"unknown stage {s!r}; choose from {ALL_STAGES}")

    # ---------------------------------------------------------------- data
    if args.synthetic:
        from .synthetic import make_synthetic
        levels, usrec = make_synthetic()
        print("[SYNTHETIC MODE — plumbing check, NOT results]")
    else:
        from . import etl
        if not config.DB_PATH.exists():
            etl.run_etl()
        levels, usrec = etl.load_clean()

    print(f"Data: {levels.index[0].date()} → {levels.index[-1].date()}, "
          f"{len(levels)} days × {levels.shape[1]} tenors")
    changes = first_diff(levels)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs: list = []

    # ---------------------------------------------------------------- core
    res_corr = fit_pca(changes, mode="corr")
    res_cov = fit_pca(changes, mode="cov")

    if "core" in stages:
        _h("1. STATIONARITY & FACTOR STRUCTURE")
        adf = adf_table(levels)
        print("ADF p-values (levels should fail to reject; diffs should reject):")
        print(adf)
        print(f"  → {(adf['p_level'] > 0.10).sum()}/{len(adf)} tenors non-stationary "
              f"in levels; {(adf['p_diff'] < 0.01).sum()}/{len(adf)} stationary in diffs")

        print(f"\nEVR (corr): {np.round(res_corr.evr[:4] * 100, 1)}%  "
              f"cum3 = {res_corr.evr[:3].sum():.1%}")
        print(f"EVR (cov):  {np.round(res_cov.evr[:4] * 100, 1)}%  "
              f"cum3 = {res_cov.evr[:3].sum():.1%}")
        print(f"sklearn cross-check, max |deviation|: "
              f"{sklearn_crosscheck(changes, res_corr):.2e}")
        print(f"Effective rank (full sample): {res_corr.effective_rank:.2f}")

        c1 = float(np.corrcoef(res_corr.scores["PC1"], changes.mean(axis=1))[0, 1])
        spread = (levels["y10"] - levels["y2"]).reindex(res_corr.scores.index)
        c2 = float(np.corrcoef(res_corr.scores["PC2"].cumsum(), spread)[0, 1])
        print(f"\nEconomic validation:")
        print(f"  corr(PC1, mean Δy)          = {c1:+.3f}   (expect ≈ 1)")
        print(f"  corr(cum PC2, 10Y–2Y spread) = {c2:+.3f}   (expect |ρ| > 0.9)")

        rmse = rmse_table(changes, res_corr)
        print(f"\nReconstruction RMSE per tenor (bps/day):")
        print(rmse)

        adf.to_csv(config.FIG_DIR / "t1_adf.csv")
        rmse.to_csv(config.FIG_DIR / "t2_rmse.csv")
        figs += [plots.scree(res_corr.evr),
                 plots.loadings(res_corr.eigenvectors),
                 plots.cumulative_scores(res_corr.scores, usrec)]

    # -------------------------------------------------------------- events
    study = run_event_study(levels, res_cov.eigenvectors)
    if "events" in stages:
        _h("2. EVENT STUDY — SIX MACRO EPISODES")
        tbl = summary_table(study)
        print(tbl)
        worst = tbl["residual_share"].max()
        print(f"\nMax residual share across episodes: {worst:.1%} "
              f"({'good — 3 factors span the moves' if worst < 0.25 else 'inspect: some episode has structure beyond PC1-3'})")
        tbl.to_csv(config.FIG_DIR / "t3_events.csv")
        figs.append(plots.episode_bars(study))

    # ------------------------------------------------------------- rolling
    if "rolling" in stages:
        _h("3. ROLLING PCA — WHEN STRUCTURE BREAKS")
        summary, roll_loads = rolling_pca(changes)
        print(f"{len(summary)} windows, PC2/PC3 swaps handled: "
              f"{summary.attrs['n_swaps']}")
        print(f"PC1 share:  median {summary['evr1'].median():.1%}, "
              f"max {summary['evr1'].max():.1%} on {summary['evr1'].idxmax().date()}")
        print(f"Eff. rank:  median {summary['erank'].median():.2f}, "
              f"min {summary['erank'].min():.2f} on {summary['erank'].idxmin().date()}")
        # --- explicit test of the crisis-spike hypothesis -----------------
        # Claim under test: "in crises the curve collapses toward one
        # dimension, so PC1's share spikes". Tested in BOTH bases, because
        # correlation-mode z-scores within each window and therefore removes
        # exactly the volatility-explosion effect the claim is about.
        crises = {"GFC 2008": ("2008-06-01", "2009-06-30"),
                  "COVID 2020": ("2020-02-01", "2020-12-31"),
                  "Hikes 2022": ("2022-03-01", "2023-07-31")}
        print("\nCrisis-spike hypothesis — PC1 share inside vs outside crises:")
        for label, mode in (("corr", "corr"), ("cov ", "cov")):
            s = summary if mode == "corr" else rolling_pca(changes, mode="cov")[0]
            mask = np.zeros(len(s), dtype=bool)
            for a, b in crises.values():
                mask |= (s.index >= a) & (s.index <= b)
            print(f"  [{label}] baseline {s.loc[~mask, 'evr1'].median():.1%} | "
                  f"crisis windows {s.loc[mask, 'evr1'].median():.1%} | "
                  f"crisis max {s.loc[mask, 'evr1'].max():.1%}")
            for label2, (a, b) in crises.items():
                w = s.loc[a:b, "evr1"]
                if len(w):
                    print(f"      {label2:11s} median {w.median():.1%}, max {w.max():.1%}")

        print("\n⚠ Overlapping windows ⇒ strongly autocorrelated series; "
              "descriptive, not a significance test.")
        summary.to_csv(config.FIG_DIR / "t4_rolling.csv")
        figs += [plots.rolling_evr(summary, usrec),
                 plots.rolling_loadings(roll_loads, usrec),
                 plots.hero_timeline(res_corr.scores, levels, usrec, study)]

    # ------------------------------------------------------------- hedging
    if "hedging" in stages:
        from . import hedging
        _h("4. HEDGING — DURATION vs PC")
        base = hedging.long_position("y10", 100e6)
        dur = hedging.duration_hedge(base, "y2")
        pch = hedging.pc_hedge(base, res_cov.eigenvectors, ("y2", "y5"))
        for pos in (base, dur, pch):
            e = pos.exposure(res_cov.eigenvectors, k=3)
            print(f"{pos.label:52s} PC1/PC2/PC3 $exposure: {np.round(e, 0)}")
        pnl = hedging.pnl_table([base, dur, pch],
                                {k: d["delta"] for k, d in study.items()})
        print("\nP&L across episodes ($):")
        print(pnl)
        pnl.to_csv(config.FIG_DIR / "t5_hedging.csv")
        figs.append(plots.hedging_comparison(pnl))

    # --------------------------------------------------------------- stats
    if "stats" in stages:
        from . import resampling
        _h("5. STATISTICAL APPENDIX")
        perm = resampling.permutation_test(changes, n_perm=args.n_resamples)
        boot = resampling.block_bootstrap_evr1(changes, n_boot=args.n_resamples)
        print(f"Observed EVR1        : {perm.observed_evr1:.1%}")
        print(f"Permutation null     : mean {perm.null_mean:.1%}, "
              f"max {perm.null_evr1.max():.1%}  (1/N = {1/changes.shape[1]:.1%})")
        print(f"  → p-value          : {perm.p_value:.4f}"
              + ("  [observed lies entirely outside null]"
                 if perm.observed_evr1 > perm.null_evr1.max() else ""))
        print(f"Block bootstrap 95%CI: [{boot.ci_lo:.1%}, {boot.ci_hi:.1%}], "
              f"block = {boot.block_length}d")

    # ------------------------------------------------------- nelson_siegel
    if "nelson_siegel" in stages:
        from . import nelson_siegel as ns
        _h("6. NELSON-SIEGEL vs PCA - CONVERGENT VALIDITY")
        ns_res = ns.fit_nelson_siegel(levels)
        print(f"NS daily fit RMSE: {ns_res.rmse_bps:.1f} bps  (lambda = {ns_res.lam})")
        corr = ns.factor_correlation(ns_res, res_corr.scores)
        print("\n|corr| between NS beta changes and PC scores:")
        print(corr.round(3))
        d = ns.diagonal_strength(corr)
        print(f"\nDiagonal |corr|: min {d['min_diagonal_abs_corr']:.2f}, "
              f"mean {d['mean_diagonal_abs_corr']:.2f}; "
              f"max off-diagonal {d['max_offdiagonal_abs_corr']:.2f}")
        print("→ slope and curvature agree strongly; level is weaker because "
              "NS forces a flat level loading while the empirical PC1 tilts "
              "toward longer maturities.")
        figs.append(plots.ns_pca_comparison(
            ns.ns_loadings(), res_corr.eigenvectors, corr))

    # --------------------------------------------------------- sensitivity
    if "sensitivity" in stages:
        from . import sensitivity
        _h("7. ROBUSTNESS CHECKS")
        out = sensitivity.run_all_sensitivity(levels)
        for name, df in out.items():
            print(f"\n--- {name} ---")
            print(df.iloc[:, :7] if df.shape[1] > 7 else df)
            df.to_csv(config.FIG_DIR / f"t6_sens_{name}.csv")
        ang = out["subperiod_angles"].max().max()
        print(f"\nMax angle between subperiod and full-sample factors: {ang:.1f}° "
              f"({'stable' if ang < 15 else 'NOTE: factors drift across periods'})")

    # ----------------------------------------------------------------- sql
    if "sql" in stages and not args.synthetic and config.DB_PATH.exists():
        from . import queries
        _h("8. SQL ANALYTICS")
        for name, df in queries.run_all_queries().items():
            print(f"\n--- {name} ---")
            print(df.head(10).to_string(index=False))
    elif "sql" in stages:
        print("\n[sql stage skipped — needs a real data/yield_data.db]")

    _h("FIGURES WRITTEN")
    for p in figs:
        print(f"  {p}")
    print(f"\nTables → {config.FIG_DIR}/*.csv")


if __name__ == "__main__":
    main()
