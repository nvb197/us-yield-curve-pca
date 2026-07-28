"""All figures. Every function saves a PNG into figures/ and returns the path.

Conventions: units labelled on every axis (pp vs bps — pitfall #3); NBER
recessions shaded grey via axvspan; correlation-mode figures for structure,
covariance-mode basis for the event study.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _shade_recessions(ax, usrec: pd.Series):
    """Grey bands for USREC==1 periods."""
    rec = usrec.astype(bool)
    edges = rec.astype(int).diff().fillna(0)
    starts = list(rec.index[edges == 1])
    ends = list(rec.index[edges == -1])
    if rec.iloc[0]:
        starts.insert(0, rec.index[0])
    if len(ends) < len(starts):
        ends.append(rec.index[-1])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="grey", alpha=0.25, zorder=0)


def scree(evr: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr * 100, color="#2b6cb0")
    ax.plot(x, np.cumsum(evr) * 100, "o-", color="#c05621", label="cumulative")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)",
           title="Scree plot — three factors carry ~95% of daily variation")
    ax.legend()
    return _save(fig, "01_scree")


def loadings(V: np.ndarray, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for j, (name, style) in enumerate(
            [("PC1 Level", "-o"), ("PC2 Slope", "-s"), ("PC3 Curvature", "-^")]):
        ax.plot(config.MATURITIES, V[:, j], style, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Maturity (years)", ylabel="Loading",
           title=f"Factor loadings across the curve{title_suffix}")
    ax.set_xscale("log")
    ax.set_xticks(config.MATURITIES)
    ax.set_xticklabels([c.upper() for c in config.COLUMNS])
    ax.legend()
    return _save(fig, "02_loadings")


def cumulative_scores(scores: pd.DataFrame, usrec: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, pc, label in zip(axes, ["PC1", "PC2", "PC3"],
                             ["Level", "Slope", "Curvature"]):
        ax.plot(scores.index, scores[pc].cumsum(), lw=1)
        _shade_recessions(ax, usrec.reindex(scores.index).ffill().fillna(0))
        ax.set_ylabel(f"cum {pc}\n({label})")
    axes[-1].set_xlabel("Grey bands = NBER recessions")
    fig.suptitle("Cumulative factor scores — 25 years of macro in three lines")
    return _save(fig, "03_cum_scores")


def rolling_evr(summary: pd.DataFrame, usrec: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.plot(summary.index, summary["evr1"] * 100, lw=1.2, color="#2b6cb0")
    ax1.set_ylabel("PC1 share of variance (%)")
    ax1.set_title("When structure breaks: rolling 252-day PCA")
    ax2.plot(summary.index, summary["erank"], lw=1.2, color="#c05621")
    ax2.set_ylabel("Effective rank\n(market dimension)")
    ax2.set_xlabel("Grey bands = NBER recessions")
    for ax in (ax1, ax2):
        _shade_recessions(ax, usrec.reindex(summary.index).ffill().fillna(0))
    return _save(fig, "04_rolling")


def episode_bars(study: dict):
    n = len(study)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.1 * ((n + 1) // 2), 6),
                             sharey=False)
    axes = np.array(axes).ravel()
    x = np.arange(len(config.COLUMNS))
    colors = {"PC1": "#2b6cb0", "PC2": "#c05621", "PC3": "#38a169"}
    for ax, (key, d) in zip(axes, study.items()):
        bottom_pos = np.zeros(len(x))
        bottom_neg = np.zeros(len(x))
        for pc, c in d["contribs"].items():
            vals = c.values * 100  # pp -> bps
            pos, neg = np.clip(vals, 0, None), np.clip(vals, None, 0)
            ax.bar(x, pos, bottom=bottom_pos, color=colors[pc], label=pc)
            ax.bar(x, neg, bottom=bottom_neg, color=colors[pc])
            bottom_pos += pos
            bottom_neg += neg
        ax.plot(x, d["delta"].values * 100, "k_", ms=12, label="actual ΔY")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in config.COLUMNS],
                           rotation=45, fontsize=6.5)
        ax.set_title(f"{d['label']}\nresid share {d['residual_share']:.0%}",
                     fontsize=7.5)
    for ax in axes[len(study):]:
        ax.axis("off")
    axes[0].set_ylabel("Contribution to ΔY (bps)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4)
    fig.suptitle("Six macro episodes decomposed into factor contributions")
    return _save(fig, "05_episodes")


def hero_timeline(scores: pd.DataFrame, levels: pd.DataFrame,
                  usrec: pd.Series, study: dict):
    """Cumulative PC2 vs 10Y-2Y spread + NBER bands + episode markers."""
    fig, ax = plt.subplots(figsize=(9, 4))
    spread = (levels["y10"] - levels["y2"]).reindex(scores.index)
    ax.plot(spread.index, spread, lw=1, color="#2b6cb0", label="10Y–2Y spread (pp)")
    ax2 = ax.twinx()
    ax2.plot(scores.index, scores["PC2"].cumsum(), lw=1, color="#c05621",
             label="cumulative PC2 (slope factor)")
    ax2.grid(False)
    _shade_recessions(ax, usrec.reindex(scores.index).ffill().fillna(0))
    ax.axhline(0, color="black", lw=0.7)
    for key, d in study.items():
        ax.axvline(pd.Timestamp(d["t0"]), color="green", lw=0.7, ls=":")
    ax.set_ylabel("10Y–2Y (pp)")
    ax2.set_ylabel("cum PC2")
    ax.set_title("The slope factor is the policy cycle: cum PC2 vs 10Y–2Y "
                 "(grey = NBER recessions, dotted = episode starts)")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)
    return _save(fig, "00_hero_timeline")


def rolling_loadings(loadings: dict, usrec: pd.Series):
    """Do the factor SHAPES themselves move over time, or only their variance
    shares? One line per tenor, per factor. Flat lines = stable structure."""
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True)
    for ax, (pc, df) in zip(axes, loadings.items()):
        for col in df.columns:
            ax.plot(df.index, df[col], lw=0.9, label=col.upper())
        _shade_recessions(ax, usrec.reindex(df.index).ffill().fillna(0))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(f"{pc} loading")
    axes[0].legend(ncol=8, fontsize=6.5, loc="upper center")
    axes[0].set_title("Do the factor shapes themselves drift? "
                      "(rolling loadings, grey = NBER recessions)")
    axes[-1].set_xlabel("Window end date")
    return _save(fig, "07_rolling_loadings")


def hedging_comparison(pnl: pd.DataFrame):
    """P&L of unhedged vs duration-hedged vs PC-hedged across episodes —
    the punchline of the risk-management section (doc E.7)."""
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x = np.arange(len(pnl))
    width = 0.8 / len(pnl.columns)
    colors = ["#718096", "#c05621", "#2b6cb0"]
    for i, col in enumerate(pnl.columns):
        ax.bar(x + i * width, pnl[col] / 1e6, width,
               label=col, color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(pnl.columns) - 1) / 2)
    ax.set_xticklabels([s[:28] for s in pnl.index], rotation=20,
                       ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("P&L ($mm)")
    ax.set_title("Duration hedge protects against parallel shocks only; "
                 "PC-hedge neutralises level AND slope")
    ax.legend(fontsize=7.5)
    return _save(fig, "08_hedging")



def ns_pca_comparison(ns_loadings_matrix, pca_V, corr_df):
    """NS parametric loadings against PCA statistical loadings, plus the
    correlation heatmap between the two methods' factors."""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = config.MATURITIES
    for j, name in enumerate(["Level", "Slope", "Curvature"]):
        ax1.plot(x, ns_loadings_matrix[:, j], "--", lw=1.5, alpha=0.9, label=f"NS {name}")
        v = pca_V[:, j]
        if v @ ns_loadings_matrix[:, j] < 0:
            v = -v
        scale = np.abs(ns_loadings_matrix[:, j]).max() / max(np.abs(v).max(), 1e-12)
        ax1.plot(x, v * scale, "o-", lw=1, ms=3, label=f"PCA {name}")
    ax1.set_xscale("log"); ax1.set_xticks(x)
    ax1.set_xticklabels([c.upper() for c in config.COLUMNS], fontsize=6.5)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_title("Parametric (NS) vs statistical (PCA) loadings")
    ax1.legend(fontsize=6.5, ncol=2)

    im = ax2.imshow(np.abs(corr_df.values), cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(corr_df.columns)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(corr_df.index, fontsize=7)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{corr_df.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(corr_df.values[i,j]) > 0.5 else "black")
    ax2.set_title("|corr|: NS beta changes vs PC scores")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    return _save(fig, "09_ns_vs_pca")
