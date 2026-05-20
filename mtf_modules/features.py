"""
mtf_modules/features.py
────────────────────────────────────────────
Feature matrix assembly.

Calls indicators.py functions per timeframe,
aligns higher-TF features to 1m index (no lookahead),
and assembles one wide master DataFrame.

Usage
-----
from mtf_modules.features import build_master, FEATURE_COLS
master = build_master(df_1m, df_spy=df_spy, info=info)
X = master[FEATURE_COLS]
"""

import numpy as np
import pandas as pd
from . import indicators as ind
from .data import fetch_resample, regular_session


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _prefix(df_feat: pd.DataFrame, pfx: str) -> pd.DataFrame:
    """Add prefix to all columns."""
    return df_feat.rename(columns={c: f"{pfx}_{c}" for c in df_feat.columns})


def _align_htf(df_1m: pd.DataFrame,
               df_htf: pd.DataFrame,
               cols: list[str]) -> pd.DataFrame:
    """
    Forward-fill higher-TF features onto 1m index.
    shift(1) avoids lookahead: each 1m bar uses the *completed* HTF bar.
    """
    shifted = df_htf[cols].shift(1)
    return shifted.reindex(df_1m.index, method="ffill")


# ──────────────────────────────────────────────
# PER-SCALE INDICATOR BLOCK
# ──────────────────────────────────────────────

def _scale_features(df: pd.DataFrame, pfx: str,
                    float_shares: int | None = None,
                    short_ratio:  float | None = None,
                    df_spy: pd.DataFrame | None = None,
                    anchor_date: str | None = None) -> pd.DataFrame:
    """
    Compute all indicators for one OHLCV DataFrame.
    Returns wide DataFrame with prefixed column names.
    """
    parts = []

    # ── Returns ──
    for n in [1, 3, 5, 10]:
        parts.append(df["close"].pct_change(n).rename(f"ret{n}"))

    # ── VWAP family ──
    v = ind.vwap(df)
    parts.append(v)
    parts.append(ind.vwap_zscore(df, v))

    if anchor_date and pfx == "m1":
        parts.append(ind.anchored_vwap(df, anchor_date))

    # ── ATR ──
    a = ind.atr(df)
    parts.append(a)
    parts.append((a / (df["close"] + 1e-9)).rename("atr_pct"))
    parts.append(((df["high"] - df["low"]) / (a + 1e-9)).rename("atr_ratio"))

    # ── Volatility regime ──
    parts.append(ind.volatility_regime(df))

    # ── RSI ──
    r = ind.rsi(df)
    parts.append(r)
    parts.append((r > 70).astype(int).rename("rsi_ob"))
    parts.append((r < 30).astype(int).rename("rsi_os"))

    # ── Bollinger ──
    bb = ind.bollinger(df)
    parts.append(bb["bb_pos"])
    parts.append(bb["bb_width"])
    squeeze_flag = (bb["bb_width"] <
                    bb["bb_width"].rolling(20, min_periods=1).mean() * 0.8
                    ).astype(int).rename("bb_squeeze")
    parts.append(squeeze_flag)

    # ── EMA cross ──
    ec = ind.ema_cross(df)
    parts.append(ec["ema_cross"])
    parts.append(ec["ema_sign"])
    parts.append(ec["ema_accel"])

    # ── MACD ──
    mc = ind.macd(df)
    parts.append(mc["macd_line"])
    parts.append(mc["macd_hist"])

    # ── Volume ──
    rv = ind.relative_volume(df)
    parts.append(rv)
    parts.append((rv > 2.5).astype(int).rename("vol_spike"))
    parts.append(ind.dollar_volume(df))
    parts.append(ind.vpt(df))

    va = ind.volume_acceleration(df)
    parts.append(va["vol_accel"])
    parts.append(va["vol_burst"])

    # ── High-beta / squeeze ──
    parts.append(ind.squeeze_pressure(df, bb["bb_width"], rv))
    parts.append(ind.gamma_proxy(df, rv))
    parts.append(ind.dtc_proxy(df, short_ratio))

    if float_shares and pfx == "m1":
        parts.append(ind.float_adjusted_volume(df, float_shares))

    # ── Candle anatomy ──
    ca = ind.candle_anatomy(df)
    for col in ca.columns:
        parts.append(ca[col])

    # ── Spread / imbalance ──
    parts.append(((df["high"] - df["low"]) / (df["close"] + 1e-9)).rename("spread"))
    imb = ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
           ).rolling(5, min_periods=1).mean().rename("imbalance")
    parts.append(imb)

    # ── Relative strength vs SPY (1m only) ──
    if df_spy is not None and not df_spy.empty and pfx == "m1":
        rs = ind.relative_strength_vs(df, df_spy)
        parts.append(rs["rs_spy"])
        parts.append(rs["rs_spy_sign"])

    # ── Combine and prefix ──
    combined = pd.concat(parts, axis=1)
    combined = _prefix(combined, pfx)
    return combined


# ──────────────────────────────────────────────
# OPENING RANGE + TIME FEATURES (1m only)
# ──────────────────────────────────────────────

def _time_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    mins = np.maximum(df_1m.index.hour * 60 + df_1m.index.minute - 570, 0)
    return pd.DataFrame({
        "time_mins"    : mins,
        "time_sin"     : np.sin(2 * np.pi * mins / 390),
        "time_cos"     : np.cos(2 * np.pi * mins / 390),
        "is_first_30m" : (mins < 30).astype(int),
        "is_power_hour": (mins > 330).astype(int),
    }, index=df_1m.index)


# ──────────────────────────────────────────────
# MASTER BUILD FUNCTION
# ──────────────────────────────────────────────

def build_master(df_1m: pd.DataFrame,
                 df_spy:     pd.DataFrame | None = None,
                 info:       dict | None = None,
                 anchor_date: str | None = None) -> pd.DataFrame:
    """
    Build the full multi-timeframe feature matrix aligned to 1m index.

    Parameters
    ----------
    df_1m        : 1-minute OHLCV (ET-localised, regular session recommended)
    df_spy       : SPY 1m for RS calculation (optional)
    info         : dict from data.get_info() — provides float_shares, short_ratio
    anchor_date  : e.g. IPO date for anchored VWAP ("YYYY-MM-DD")

    Returns
    -------
    master : wide DataFrame with all features + raw OHLCV columns
             Index = same as df_1m
    """
    info         = info or {}
    float_shares = info.get("float_shares")
    short_ratio  = info.get("short_ratio")

    # ── Resample to higher timeframes ──
    df_5m  = fetch_resample(df_1m, "5min")
    df_15m = fetch_resample(df_1m, "15min")
    df_30m = fetch_resample(df_1m, "30min")

    # ── Indicators per scale ──
    feat_1m  = _scale_features(df_1m,  "m1",
                                float_shares=float_shares,
                                short_ratio=short_ratio,
                                df_spy=df_spy,
                                anchor_date=anchor_date)
    feat_5m  = _scale_features(df_5m,  "m5")
    feat_15m = _scale_features(df_15m, "m15")
    feat_30m = _scale_features(df_30m, "m30")

    # ── Align HTF → 1m (no lookahead) ──
    master = df_1m.copy()
    master = pd.concat([master, feat_1m], axis=1)

    for feat_htf, pfx in [(feat_5m, "m5"), (feat_15m, "m15"), (feat_30m, "m30")]:
        cols    = list(feat_htf.columns)
        aligned = _align_htf(df_1m, feat_htf, cols)
        master  = pd.concat([master, aligned], axis=1)

    # ── Opening range ──
    or_feat = ind.opening_range(df_1m)
    master  = pd.concat([master, or_feat], axis=1)

    # ── Time features ──
    tf = _time_features(df_1m)
    master = pd.concat([master, tf], axis=1)

    master.dropna(inplace=True)
    return master


def get_feature_cols(master: pd.DataFrame) -> list[str]:
    """Return all engineered feature column names (excludes raw OHLCV and label)."""
    exclude = {"open", "high", "low", "close", "volume", "label"}
    return [c for c in master.columns if c not in exclude]


# ──────────────────────────────────────────────
# MARKETCAP TIER HELPER
# ──────────────────────────────────────────────

def marketcap_tier(market_cap: int | None) -> dict:
    """
    Classify market cap into tier and return recommended
    model hyperparameter adjustments.

    Returns dict with: tier, atr_mult, min_leaf, depth, label.
    """
    if market_cap is None:
        return {"tier": "unknown", "atr_mult": 1.0,
                "min_leaf": 10, "depth": 6, "label": "Unknown"}

    if   market_cap < 300_000_000:
        return {"tier": "micro",  "atr_mult": 2.0,
                "min_leaf": 20, "depth": 4, "label": "Micro Cap  (<$300M)"}
    elif market_cap < 2_000_000_000:
        return {"tier": "small",  "atr_mult": 1.5,
                "min_leaf": 15, "depth": 5, "label": "Small Cap  ($300M–$2B)"}
    elif market_cap < 10_000_000_000:
        return {"tier": "mid",    "atr_mult": 1.0,
                "min_leaf": 10, "depth": 6, "label": "Mid Cap    ($2B–$10B)"}
    else:
        return {"tier": "large",  "atr_mult": 0.6,
                "min_leaf": 8,  "depth": 6, "label": "Large Cap  (>$10B)"}
