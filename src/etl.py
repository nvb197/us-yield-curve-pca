"""ETL: FRED API -> validate -> clean -> SQLite.

Cleaning rule (doc 2.3): inner-join the core tenor set, DROP any day with a
missing tenor (loss <1%), optionally forward-fill isolated 1-day gaps only
(limit=1) and log how many. Blanket ffill is forbidden: asymmetric fills
create artificial zero-changes in some columns only, which shrinks their
variance and distorts the correlation structure -> distorts eigenvectors.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
from fredapi import Fred

from . import config


# --- Fetch ---------------------------------------------------------------

def fetch_yields() -> pd.DataFrame:
    """Fetch all tenor series from FRED. Wide frame indexed by date."""
    key = config.fred_api_key()
    if not key:
        raise RuntimeError(
            f"Set the {config.FRED_API_KEY_ENV} environment variable "
            "(free key: https://fred.stlouisfed.org)."
        )
    fred = Fred(api_key=key)
    frame = {}
    for code, (col, _) in config.TENORS.items():
        s = fred.get_series(code, observation_start=config.START_DATE)
        frame[col] = pd.to_numeric(s, errors="coerce")  # FRED "." -> NaN
    df = pd.DataFrame(frame)
    df.index.name = "date"
    return df[config.COLUMNS]


def fetch_recessions() -> pd.Series:
    """NBER recession indicator (monthly, binary)."""
    fred = Fred(api_key=config.fred_api_key())
    s = fred.get_series(config.RECESSION_SERIES,
                        observation_start=config.START_DATE)
    s.name = "usrec"
    s.index.name = "date"
    return s.astype(int)


# --- Clean ---------------------------------------------------------------

def clean_yields(raw: pd.DataFrame, ffill_limit: int = 0) -> tuple[pd.DataFrame, dict]:
    """ffill_limit=0 by default: market holidays appear in FRED as rows where
    ALL tenors are NaN. Forward-filling them would inject ~290 artificial
    'zero-change' days over 25 years (~4% of the sample), deflating measured
    volatility and adding no information. Dropping them is standard practice
    and matches the stated methodology — the cost is that a few daily changes
    span a long weekend. Pass ffill_limit=1 to reproduce the older behaviour.

    Returns (clean_frame, log_dict)."""
    df = raw.sort_index()
    n_raw = len(df)
    if ffill_limit:
        before = df.isna().sum().sum()
        df = df.ffill(limit=ffill_limit)
        n_ffilled = int(before - df.isna().sum().sum())
    else:
        n_ffilled = 0
    clean = df.dropna(how="any")
    log = dict(
        n_raw=n_raw,
        n_dropped=n_raw - len(clean),
        n_ffilled=n_ffilled,
        pct_dropped=round(100 * (n_raw - len(clean)) / max(n_raw, 1), 3),
    )
    assert not clean.isna().any().any()
    assert clean.index.is_monotonic_increasing
    return clean, log


def recession_daily(usrec_monthly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Monthly USREC -> daily. The one place where ffill IS correct: a month's
    recession flag applies to every day of that month."""
    return (usrec_monthly.reindex(usrec_monthly.index.union(daily_index))
            .ffill().reindex(daily_index).fillna(0).astype(int))


# --- SQLite --------------------------------------------------------------

def write_db(raw: pd.DataFrame, clean: pd.DataFrame,
             usrec: pd.Series, log: dict, db_path=None) -> None:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        raw.reset_index().melt(id_vars="date", var_name="tenor",
                               value_name="value").to_sql(
            "yields_raw", con, if_exists="replace", index=False)
        clean.reset_index().to_sql("yields_clean", con,
                                   if_exists="replace", index=False)
        usrec.reset_index().to_sql("recessions", con,
                                   if_exists="replace", index=False)
        pd.DataFrame([{**log, "run_ts": datetime.now(timezone.utc).isoformat(),
                       "notes": "core tenor set, rule 2.3"}]).to_sql(
            "etl_log", con, if_exists="append", index=False)


def load_clean(db_path=None) -> tuple[pd.DataFrame, pd.Series]:
    db_path = db_path or config.DB_PATH
    with sqlite3.connect(db_path) as con:
        clean = pd.read_sql("SELECT * FROM yields_clean", con,
                            parse_dates=["date"]).set_index("date")
        rec = pd.read_sql("SELECT * FROM recessions", con,
                          parse_dates=["date"]).set_index("date")["usrec"]
    return clean[config.COLUMNS], rec


def run_etl() -> None:
    raw = fetch_yields()
    clean, log = clean_yields(raw)
    usrec = recession_daily(fetch_recessions(), clean.index)
    write_db(raw, clean, usrec, log)
    print(f"ETL ok: {len(clean)} days, dropped {log['pct_dropped']}%, "
          f"ffilled {log['n_ffilled']} cells -> {config.DB_PATH}")