"""Central configuration: tenor set, dates, paths, episode anchors.

Design decisions (see README "Data & design decisions"):
- Core tenor set = {3M..10Y}: continuous coverage since 2000, no structural gaps.
- 1M excluded (starts 2001-07, money-market noise). 30Y excluded (discontinued
  2002-02 -> 2006-02). 20Y excluded (gap 1987-1993, redundant with 10Y/30Y).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "yield_data.db"
FIG_DIR = ROOT / "figures"

# --- FRED ----------------------------------------------------------------
# Never hardcode the key. Get a free one at https://fred.stlouisfed.org
FRED_API_KEY_ENV = "FRED_API_KEY"


def fred_api_key() -> str | None:
    return os.environ.get(FRED_API_KEY_ENV)


# FRED series code -> (column name, maturity in years). Order = maturity order;
# code relies on columns being sorted short -> long (sign conventions, plots).
TENORS: dict[str, tuple[str, float]] = {
    "DGS3MO": ("m3", 0.25),
    "DGS6MO": ("m6", 0.50),
    "DGS1":   ("y1", 1.0),
    "DGS2":   ("y2", 2.0),
    "DGS3":   ("y3", 3.0),
    "DGS5":   ("y5", 5.0),
    "DGS7":   ("y7", 7.0),
    "DGS10":  ("y10", 10.0),
}
COLUMNS = [c for c, _ in TENORS.values()]          # ["m3", ..., "y10"]
MATURITIES = [m for _, m in TENORS.values()]        # [0.25, ..., 10.0]
RECESSION_SERIES = "USREC"                          # NBER, monthly, binary

START_DATE = "2000-01-01"

# --- PCA / rolling -------------------------------------------------------
N_FACTORS = 3
ROLL_WINDOW = 252   # 1 trading year; sensitivity: 126 / 504
ROLL_STEP = 21      # 1 trading month

# --- Event study (section 5.4 of the project doc) ------------------------
# Windows are anchored EX ANTE to external, documented events (FOMC decisions,
# public announcements) -- never chosen by looking at the chart. Each entry:
# key -> (t0, t1 or None=latest, label, expected signature, anchor sources).
EPISODES: dict[str, dict] = {
    "hiking_2004": dict(
        t0="2004-06-30", t1="2006-06-29",
        label="2004-06 Fed hikes ('Greenspan's conundrum')",
        expected="Bear flattening — PC2 dominant, long end barely moves",
        anchors="t0 = FOMC first hike (1.00->1.25); t1 = FOMC last hike (5.25)",
    ),
    "gfc_2008": dict(
        t0="2007-09-18", t1="2008-12-16",
        label="2007-08 GFC easing to zero",
        expected="Bull steepening + level collapse; rolling PC1-EVR spike",
        anchors="t0 = FOMC first cut (-50bp); t1 = FOMC ZIRP (0-0.25%)",
    ),
    "taper_2013": dict(
        t0="2013-05-22", t1="2013-09-05",
        label="2013 Taper tantrum",
        expected="Bear steepening — long end leads the selloff",
        anchors="t0 = Bernanke JEC testimony; t1 = eve of Sep FOMC (10Y peak)",
    ),
    "covid_2020": dict(
        t0="2020-02-19", t1="2020-03-23",
        label="2020 COVID crash to zero",
        expected="Violent level collapse; market ~1-dimensional",
        anchors="t0 = pre-crisis equity peak; t1 = Fed unlimited-QE announcement",
    ),
    "inflation_2022": dict(
        t0="2022-03-16", t1="2023-07-26",
        label="2022-23 inflation fight (deepest inversion since 1981)",
        expected="Textbook bear flattening — PC2 share unusually high",
        anchors="t0 = FOMC first hike; t1 = FOMC last hike (5.25-5.50)",
    ),
    "easing_2024": dict(
        t0="2024-09-18", t1=None,
        label="2024- easing cycle / dis-inversion",
        expected="Bull steepening — PC2 reverses sign",
        anchors="t0 = FOMC first cut (-50bp); t1 = latest observation",
    ),
}
