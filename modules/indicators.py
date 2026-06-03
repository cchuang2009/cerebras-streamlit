"""
mtf_modules/indicators.py
────────────────────────────────────────────
Pure technical indicator functions.
Each function accepts a DataFrame and returns a named pd.Series.
No side-effects; easy to unit-test independently.

Usage
-----
from mtf_modules.indicators import (
    vwap, atr, rsi, bollinger, ema_cross,
    vpt, anchored_vwap, gamma_proxy,
    squeeze_pressure, dtc_proxy,
)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# CLASSIC INDICATORS
# ──────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP — resets each calendar day.
    Requires: high, low, close, volume.
    """
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    dk  = df.index.normalize()
    cum_tpv = (tp * df["volume"]).groupby(dk).cumsum()
    cum_vol = df["volume"].groupby(dk).cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).rename("vwap")


def anchored_vwap(df: pd.DataFrame,
                  anchor_date: str | pd.Timestamp) -> pd.Series:
    """
    Anchored VWAP — cumulative VWAP from a specific anchor date.
    Useful for IPO day, earnings day, or major pivot anchor.

    Parameters
    ----------
    anchor_date : "YYYY-MM-DD" or pd.Timestamp
    """
    anchor = pd.Timestamp(anchor_date)
    # tz-aware comparison
    if df.index.tzinfo is not None and anchor.tzinfo is None:
        import pytz
        anchor = pytz.timezone("America/New_York").localize(anchor)

    sub = df[df.index >= anchor].copy()
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3
    cum_tpv = (tp * sub["volume"]).cumsum()
    cum_vol = sub["volume"].cumsum()
    result  = (cum_tpv / (cum_vol + 1e-9)).rename("avwap")
    return result.reindex(df.index)   # NaN before anchor


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    c, h, l = df["close"], df["high"], df["low"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().rename(f"atr{period}")


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    return (100 - 100 / (1 + gain / (loss + 1e-9))).rename(f"rsi{period}")


def bollinger(df: pd.DataFrame,
              period: int = 20,
              std_mult: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: bb_mid, bb_upper, bb_lower,
                                    bb_pos (0=lower,1=upper),
                                    bb_width (band width / mid).
    """
    c    = df["close"]
    mid  = c.rolling(period, min_periods=1).mean()
    std  = c.rolling(period, min_periods=1).std().fillna(0)
    up   = mid + std_mult * std
    lo   = mid - std_mult * std
    rng  = up - lo + 1e-9
    return pd.DataFrame({
        "bb_mid"  : mid,
        "bb_upper": up,
        "bb_lower": lo,
        "bb_pos"  : (c - lo) / rng,
        "bb_width": rng / (mid + 1e-9),
    }, index=df.index)


def ema_cross(df: pd.DataFrame,
              fast: int = 9,
              slow: int = 21) -> pd.DataFrame:
    """
    EMA crossover features.
    Returns: ema_fast, ema_slow, ema_cross (normalised gap),
             ema_sign (+1/-1), ema_accel (1st diff of cross).
    """
    c    = df["close"]
    ef   = c.ewm(span=fast,  adjust=False).mean()
    es   = c.ewm(span=slow,  adjust=False).mean()
    gap  = (ef - es) / (c + 1e-9)
    return pd.DataFrame({
        f"ema{fast}"      : ef,
        f"ema{slow}"      : es,
        "ema_cross"       : gap,
        "ema_sign"        : np.sign(gap),
        "ema_accel"       : gap.diff(),
    }, index=df.index)


def macd(df: pd.DataFrame,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    c    = df["close"]
    ef   = c.ewm(span=fast,   adjust=False).mean()
    es   = c.ewm(span=slow,   adjust=False).mean()
    line = ef - es
    sig  = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd_line"  : line,
        "macd_signal": sig,
        "macd_hist"  : line - sig,
    }, index=df.index)


# ──────────────────────────────────────────────
# VOLUME INDICATORS
# ──────────────────────────────────────────────

def relative_volume(df: pd.DataFrame,
                    period: int = 20) -> pd.Series:
    """RVOL = current volume / N-bar rolling mean."""
    return (df["volume"] / (df["volume"].rolling(period, min_periods=1).mean() + 1e-9)
            ).rename("rvol")


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """Dollar volume = close × volume. Normalises across price levels."""
    return (df["close"] * df["volume"]).rename("dollar_vol")


def vpt(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Price Trend — cumulative volume adjusted by % price change.
    Divergence from price signals trend exhaustion.
    """
    ret = df["close"].pct_change().fillna(0)
    return (ret * df["volume"]).cumsum().rename("vpt")


def volume_acceleration(df: pd.DataFrame,
                        s: int = 5,
                        m: int = 20,
                        l: int = 60) -> pd.DataFrame:
    """
    Two-stage volume acceleration.
    vol_accel = (fast/mid) / (mid/slow)
    vol_burst = 1 if vol_accel > 2.0
    """
    v   = df["volume"]
    vms = v.rolling(s, min_periods=1).mean()
    vmm = v.rolling(m, min_periods=1).mean()
    vml = v.rolling(l, min_periods=1).mean()
    acc = (vms / (vmm + 1e-9)) / (vmm / (vml + 1e-9))
    return pd.DataFrame({
        "vol_accel": acc,
        "vol_burst": (acc > 2.0).astype(int),
    }, index=df.index)


def float_adjusted_volume(df: pd.DataFrame,
                          float_shares: int | None) -> pd.Series:
    """
    Float Utilization = volume / float_shares.
    > 0.5  → 50%+ of float traded in one bar (extreme activity)
    > 1.0  → full float turned over (IPO-day level event)
    Returns zeros if float_shares is None.
    """
    if float_shares and float_shares > 0:
        return (df["volume"] / float_shares).rename("float_util")
    return pd.Series(0.0, index=df.index, name="float_util")


# ──────────────────────────────────────────────
# HIGH-BETA / SQUEEZE INDICATORS
# ──────────────────────────────────────────────

def vwap_zscore(df: pd.DataFrame,
                vwap_series: pd.Series,
                window: int = 20) -> pd.Series:
    """
    Z-score of (close - VWAP) / close.
    Measures stretch from intraday fair value.
    High positive → overbought vs VWAP; large negative → oversold.
    """
    dist  = (df["close"] - vwap_series) / (df["close"] + 1e-9)
    sigma = dist.rolling(window, min_periods=5).std() + 1e-9
    return (dist / sigma).rename("vwap_zscore")


def squeeze_pressure(df: pd.DataFrame,
                     bb_width: pd.Series,
                     rvol_series: pd.Series) -> pd.Series:
    """
    Composite short-squeeze pressure proxy (no external short data needed).
    Formula: (RVOL × positive_return) / BB_width
    High value = volume surge + upward momentum + compressed volatility.
    """
    ret_5 = df["close"].pct_change(5).clip(lower=0)
    raw   = (rvol_series * ret_5) / (bb_width + 1e-9)
    return raw.rolling(5, min_periods=1).mean().rename("squeeze_pressure")


def gamma_proxy(df: pd.DataFrame,
                rvol_series: pd.Series,
                window: int = 5) -> pd.Series:
    """
    Gamma squeeze proxy: high volume + shrinking bar range.
    Market makers absorbing delta-hedge buying compress candle range
    while volume stays elevated.
    """
    bar_range = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    raw       = rvol_series / (bar_range + 1e-9)
    return raw.rolling(window, min_periods=1).mean().rename("gamma_proxy")


def dtc_proxy(df: pd.DataFrame,
              short_ratio: float | None = None,
              window: int = 5) -> pd.Series:
    """
    Days-to-Cover proxy.
    If short_ratio (from yfinance info) is available, broadcast it.
    Otherwise estimate from RVOL inversion: low RVOL after a run-up
    means exit door is crowded.

    Returns a Series named "dtc_proxy".
    """
    if short_ratio is not None:
        return pd.Series(float(short_ratio), index=df.index, name="dtc_proxy")
    # Estimated: ratio of recent volume to 5-day average
    adtv = df["volume"].rolling(window, min_periods=1).mean()
    return (adtv / (df["volume"] + 1e-9)).rename("dtc_proxy")


def volatility_regime(df: pd.DataFrame,
                      fast: int = 20,
                      slow: int = 60) -> pd.Series:
    """
    Volatility regime classification:
      0 = low  (vol < 80% of slow average)
      1 = mid  (normal)
      2 = high-burst (vol > 150% of slow average)
    """
    vol_fast = df["close"].pct_change().rolling(fast, min_periods=5).std()
    vol_slow = vol_fast.rolling(slow, min_periods=10).mean()
    ratio    = vol_fast / (vol_slow + 1e-9)
    regime   = pd.cut(ratio,
                      bins=[0, 0.8, 1.5, np.inf],
                      labels=[0, 1, 2]).astype(float).fillna(1)
    return regime.rename("vol_regime")


def opening_range(df: pd.DataFrame,
                  minutes: int = 15) -> pd.DataFrame:
    """
    Opening Range features (first N minutes after 09:30).
    Returns: or_high, or_low, or_pos, or_breakout, or_breakdown.
    """
    f = df.copy()
    f["_date"] = f.index.normalize()
    f["_mins"] = f.index.hour * 60 + f.index.minute - 570

    or_h = (f[f["_mins"] < minutes]
            .groupby("_date")["high"].max().rename("_or_h"))
    or_l = (f[f["_mins"] < minutes]
            .groupby("_date")["low"].min().rename("_or_l"))
    f = f.join(or_h, on="_date").join(or_l, on="_date")

    rng = f["_or_h"] - f["_or_l"] + 1e-9
    result = pd.DataFrame({
        "or_pos"      : (f["close"] - f["_or_l"]) / rng,
        "or_breakout" : (f["close"] > f["_or_h"]).astype(int),
        "or_breakdown": (f["close"] < f["_or_l"]).astype(int),
    }, index=df.index)
    return result


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle body / wick decomposition.
    Returns: body_ratio, upper_wick, lower_wick, is_bull.
    """
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    body  = (c - o).abs()
    rng   = h - l + 1e-9
    hi_end = pd.concat([c, o], axis=1).max(axis=1)
    lo_end = pd.concat([c, o], axis=1).min(axis=1)
    return pd.DataFrame({
        "body_ratio" : body / rng,
        "upper_wick" : (h - hi_end) / rng,
        "lower_wick" : (lo_end - l) / rng,
        "is_bull"    : (c > o).astype(int),
    }, index=df.index)


def relative_strength_vs(df_ticker: pd.DataFrame,
                         df_bench: pd.DataFrame,
                         window: int = 10) -> pd.DataFrame:
    """
    Relative strength of ticker vs benchmark (e.g. SPY).
    Aligns bench to ticker index via ffill.
    Returns: rs_raw (rolling mean), rs_sign (+1/-1).
    """
    bench_ret  = df_bench["close"].pct_change()
    bench_al   = bench_ret.reindex(df_ticker.index, method="ffill").fillna(0)
    ticker_ret = df_ticker["close"].pct_change().fillna(0)
    rs         = (ticker_ret - bench_al).rolling(window, min_periods=3).mean()
    return pd.DataFrame({
        "rs_spy"     : rs,
        "rs_spy_sign": np.sign(rs),
    }, index=df_ticker.index)
