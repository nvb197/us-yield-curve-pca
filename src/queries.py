"""Analytical SQL layer — demonstrates real SQL usage on top of yield_data.db,
not just to_sql()/read_sql() as a dumb store (see chat: "SQLite vs online DB").

Every function here does the aggregation IN SQL (GROUP BY, window functions,
CASE) rather than pulling raw rows into pandas and computing there. Open the
same file in DBeaver / DB Browser for SQLite to run these queries by hand and
sanity-check the results visually.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import config


def _conn(db_path=None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or config.DB_PATH)


# --- 1. Yearly summary: avg level, avg spread, inversion days -----------

YEARLY_SUMMARY_SQL = """
SELECT
    CAST(strftime('%Y', date) AS INTEGER)      AS year,
    ROUND(AVG(y10), 3)                          AS avg_10y,
    ROUND(AVG(y2),  3)                          AS avg_2y,
    ROUND(AVG(y10 - y2), 3)                     AS avg_10y_2y_spread,
    ROUND(MIN(y10 - y2), 3)                     AS min_spread,
    SUM(CASE WHEN y10 - y2 < 0 THEN 1 ELSE 0 END)      AS inverted_days,
    COUNT(*)                                    AS trading_days,
    ROUND(100.0 * SUM(CASE WHEN y10 - y2 < 0 THEN 1 ELSE 0 END)
          / COUNT(*), 1)                        AS pct_inverted
FROM yields_clean
GROUP BY year
ORDER BY year;
"""


def yearly_summary(db_path=None) -> pd.DataFrame:
    """One row per year: average level, average/min 10Y-2Y spread, and the
    % of trading days the curve spent inverted. All computed in SQL."""
    with _conn(db_path) as con:
        return pd.read_sql(YEARLY_SUMMARY_SQL, con)


# --- 2. Longest inversion streak (SQL window functions) ------------------

LONGEST_INVERSION_SQL = """
WITH flagged AS (
    SELECT date, (y10 - y2 < 0) AS inverted
    FROM yields_clean
),
grouped AS (
    SELECT date, inverted,
           ROW_NUMBER() OVER (ORDER BY date)
           - ROW_NUMBER() OVER (PARTITION BY inverted ORDER BY date) AS grp
    FROM flagged
)
SELECT MIN(date) AS start_date, MAX(date) AS end_date,
       COUNT(*)  AS trading_days_inverted
FROM grouped
WHERE inverted = 1
GROUP BY grp
ORDER BY trading_days_inverted DESC
LIMIT 5;
"""


def longest_inversion_streaks(db_path=None) -> pd.DataFrame:
    """Top-5 longest continuous inversion streaks, found with the classic
    'gaps and islands' SQL window-function pattern (ROW_NUMBER trick):
    subtracting a same-group row number from a global row number produces a
    constant `grp` for every consecutive run of the same flag value."""
    with _conn(db_path) as con:
        return pd.read_sql(LONGEST_INVERSION_SQL, con)


# --- 3. Regime classification per day (CASE on daily changes) -----------

REGIME_SQL = """
WITH diffs AS (
    SELECT date,
           y2  - LAG(y2)  OVER (ORDER BY date) AS d2,
           y10 - LAG(y10) OVER (ORDER BY date) AS d10
    FROM yields_clean
)
SELECT
    -- Clean 2x2 (doc E.5): bull/bear = direction of the overall level move
    -- (average of the two tenors); steepening/flattening = change in the
    -- 10Y-2Y spread. Independent axes, so no branch ordering bias.
    -- 'flat day' = moves too small to classify (<1bp on both axes).
    CASE
        WHEN d2 IS NULL THEN NULL
        WHEN ABS((d2 + d10) / 2.0) < 0.01 AND ABS(d10 - d2) < 0.01 THEN 'flat_day'
        WHEN (d2 + d10) / 2.0 >= 0 AND (d10 - d2) >= 0 THEN 'bear_steepening'
        WHEN (d2 + d10) / 2.0 >= 0 AND (d10 - d2) <  0 THEN 'bear_flattening'
        WHEN (d2 + d10) / 2.0 <  0 AND (d10 - d2) >= 0 THEN 'bull_steepening'
        ELSE 'bull_flattening'
    END AS regime,
    COUNT(*) AS n_days
FROM diffs
WHERE d2 IS NOT NULL
GROUP BY regime
ORDER BY n_days DESC;
"""


def regime_day_counts(db_path=None) -> pd.DataFrame:
    """Classify every day into one of the 4 curve regimes (doc E.5) using
    LAG() window function + CASE, entirely in SQL. A quick sanity check:
    'mixed' should be the largest bucket in most years (small ambiguous
    moves), with clear regime days concentrated around the 6 episodes."""
    with _conn(db_path) as con:
        return pd.read_sql(REGIME_SQL, con)


# --- 4. ETL audit trail -----------------------------------------------

def etl_audit_log(db_path=None) -> pd.DataFrame:
    """Every ETL run ever logged — reproducibility trail (doc G.6)."""
    with _conn(db_path) as con:
        return pd.read_sql(
            "SELECT * FROM etl_log ORDER BY run_ts DESC;", con)


def run_all_queries(db_path=None) -> dict[str, pd.DataFrame]:
    return dict(
        yearly=yearly_summary(db_path),
        inversions=longest_inversion_streaks(db_path),
        regimes=regime_day_counts(db_path),
        etl_log=etl_audit_log(db_path),
    )


if __name__ == "__main__":
    for name, df in run_all_queries().items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
