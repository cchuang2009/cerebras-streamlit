"""
mtf_modules/prophet_model.py
────────────────────────────────────────────
Facebook Prophet wrapper optimised for intraday high-beta stocks.

Usage
-----
from mtf_modules.prophet_model import fit_prophet, prophet_signal
result = fit_prophet(df_1m, periods=60)
"""

import logging
import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def fit_prophet(df_1m: pd.DataFrame,
                periods:        int   = 60,
                interval_width: float = 0.80,
                cache_key:      str   = "") -> dict | None:
    """
    Fit Prophet on regular-session 1m close prices.

    Parameters
    ----------
    df_1m          : 1m OHLCV (ET-localised, regular session)
    periods        : number of future 1m bars to forecast
    interval_width : confidence interval width (0.5 – 0.95)
    cache_key      : unused internally; passed by caller for cache busting

    Returns
    -------
    dict with keys:
        fc_hist, fc_fut, actual_df,
        last_actual, next_yhat, next_yhat_lo, next_yhat_hi,
        trend_chg_pct, prophet_signal, periods, interval_width
    None if insufficient data or Prophet not installed.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return None

    # ── Regular session only ──
    reg = df_1m[
        (df_1m.index.time >= pd.Timestamp("09:30").time()) &
        (df_1m.index.time <= pd.Timestamp("16:00").time())
    ].copy()

    if len(reg) < 60:
        return None

    # ── Build Prophet DataFrame ──
    vol_norm = reg["volume"].values / (reg["volume"].mean() + 1e-9)
    prophet_df = pd.DataFrame({
        "ds"         : reg.index.tz_localize(None),
        "y"          : reg["close"].values,
        "volume_norm": vol_norm,
    }).dropna()

    # ── Model — high-beta tuned ──
    m = Prophet(
        changepoint_prior_scale  = 0.8,    # high flexibility for fast-moving stocks
        changepoint_range        = 0.95,   # allow changepoints near end of data
        seasonality_prior_scale  = 0.05,   # dampen noisy seasonality
        interval_width           = interval_width,
        daily_seasonality        = False,
        weekly_seasonality       = False,
        yearly_seasonality       = False,
        n_changepoints           = 30,
    )
    # Intraday seasonality: 390-min trading day
    m.add_seasonality(
        name          = "intraday",
        period        = 390 / (60 * 24),
        fourier_order = 8,
    )
    m.add_regressor("volume_norm", standardize=True)
    m.fit(prophet_df)

    # ── Future DataFrame ──
    future               = m.make_future_dataframe(periods=periods, freq="1min")
    future["volume_norm"]= float(vol_norm[-20:].mean())
    forecast             = m.predict(future)

    # ── Split historical fit vs forecast ──
    n_hist   = len(prophet_df)
    fc_hist  = forecast.iloc[:n_hist].copy()
    fc_fut   = forecast.iloc[n_hist:].copy()

    # ── ET timestamps for display ──
    fc_hist["ds_et"] = pd.to_datetime(fc_hist["ds"]).dt.tz_localize(ET)
    fc_fut["ds_et"]  = pd.to_datetime(fc_fut["ds"]).dt.tz_localize(ET)

    last_actual  = float(prophet_df["y"].iloc[-1])
    next_yhat    = float(fc_fut["yhat"].iloc[0])    if len(fc_fut) else last_actual
    next_yhat_lo = float(fc_fut["yhat_lower"].iloc[0]) if len(fc_fut) else last_actual
    next_yhat_hi = float(fc_fut["yhat_upper"].iloc[0]) if len(fc_fut) else last_actual

    trend_chg = (next_yhat - last_actual) / (last_actual + 1e-9)
    if   trend_chg >  0.003: signal = "📈 UP"
    elif trend_chg < -0.003: signal = "📉 DOWN"
    else:                    signal = "➡️  FLAT"

    return {
        "fc_hist"       : fc_hist,
        "fc_fut"        : fc_fut,
        "actual_df"     : prophet_df,
        "last_actual"   : last_actual,
        "next_yhat"     : next_yhat,
        "next_yhat_lo"  : next_yhat_lo,
        "next_yhat_hi"  : next_yhat_hi,
        "trend_chg_pct" : trend_chg * 100,
        "prophet_signal": signal,
        "periods"       : periods,
        "interval_width": interval_width,
    }


def prophet_signal(result: dict | None) -> str:
    """Extract plain signal string from fit_prophet result."""
    if result is None:
        return "N/A"
    return result.get("prophet_signal", "N/A")


def model_agreement(catboost_trend: str,
                    prophet_signal: str) -> dict:
    """
    Compare CatBoost and Prophet direction.

    Returns dict with: agree (bool), message, color.
    """
    cb = catboost_trend.upper()
    ph = prophet_signal

    agree = (
        ("BREAKOUT" in cb and "UP"   in ph) or
        ("CRASH"    in cb and "DOWN" in ph) or
        ("SQUEEZE"  in cb and "FLAT" in ph)
    )

    if "UNCERTAIN" in cb:
        return {"agree": None,
                "message": "⚪ CatBoost UNCERTAIN — defer to Prophet",
                "color": "#7a7060"}
    if agree:
        return {"agree": True,
                "message": "✅ Both models AGREE — signal strengthened",
                "color": "#1a7a3f"}
    return {"agree": False,
            "message": "⚠️ Models DISAGREE — exercise caution",
            "color": "#b8860b"}
