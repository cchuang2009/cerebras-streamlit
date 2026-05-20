"""
mtf_modules/data.py
────────────────────────────────────────────
Data fetching layer.
All functions return clean, ET-localised DataFrames.

Usage
-----
from mtf_modules.data import fetch_1m, fetch_spy_1m, fetch_daily, get_info
"""

import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime

ET = pytz.timezone("America/New_York")


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and convert index to ET timezone."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df.sort_index()


# ──────────────────────────────────────────────
# PUBLIC FETCH FUNCTIONS
# ──────────────────────────────────────────────

def fetch_1m(ticker: str, start: str, end: str,
             prepost: bool = True) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars.

    Parameters
    ----------
    ticker  : e.g. "AAOI"
    start   : "YYYY-MM-DD"
    end     : "YYYY-MM-DD"  (exclusive — use tomorrow for today's data)
    prepost : include pre/after-market bars

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume],
    index = DatetimeTZDtype[ns, America/New_York]
    """
    df = yf.download(
        ticker, start=start, end=end,
        interval="1m", progress=False,
        auto_adjust=True, multi_level_index=False,
        prepost=prepost,
    )
    return _clean(df)


def fetch_spy_1m(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY 1-minute bars for relative-strength calculation."""
    return fetch_1m("SPY", start, end, prepost=False)


def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars."""
    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", progress=False,
        auto_adjust=True, multi_level_index=False,
    )
    return _clean(df)


def fetch_resample(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1m DataFrame to any higher timeframe in-memory.

    Parameters
    ----------
    rule : pandas offset string — "5min", "15min", "30min", "1h"
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    available = {k: v for k, v in agg.items() if k in df_1m.columns}
    return (
        df_1m[list(available)]
        .resample(rule, label="left", closed="left")
        .agg(available)
        .dropna()
    )


def get_info(ticker: str) -> dict:
    """
    Fetch fundamental info for a ticker.

    Returns dict with at minimum:
        market_cap      : int | None
        float_shares    : int | None
        shares_outstanding : int | None
        sector          : str | None
        short_percent_float : float | None   (0–1)
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "market_cap"          : info.get("marketCap"),
            "float_shares"        : info.get("floatShares"),
            "shares_outstanding"  : info.get("sharesOutstanding"),
            "sector"              : info.get("sector"),
            "short_percent_float" : info.get("shortPercentOfFloat"),
            "short_ratio"         : info.get("shortRatio"),   # days-to-cover
            "avg_volume"          : info.get("averageVolume"),
            "avg_volume_10d"      : info.get("averageVolume10days"),
        }
    except Exception:
        return {k: None for k in [
            "market_cap", "float_shares", "shares_outstanding",
            "sector", "short_percent_float", "short_ratio",
            "avg_volume", "avg_volume_10d",
        ]}


def classify_session(ts: pd.Timestamp) -> str:
    """Classify a bar into premarket / regular / afterhours / overnight."""
    t = ts.time()
    if   t < pd.Timestamp("04:00").time(): return "overnight"
    elif t < pd.Timestamp("09:30").time(): return "premarket"
    elif t < pd.Timestamp("16:00").time(): return "regular"
    else:                                   return "afterhours"


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to regular session bars only (09:30–16:00 ET)."""
    return df[
        (df.index.time >= pd.Timestamp("09:30").time()) &
        (df.index.time <= pd.Timestamp("16:00").time())
    ].copy()
