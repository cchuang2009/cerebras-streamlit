"""
mtf_modules/ipo_features.py
────────────────────────────────────────────
IPO 高 Beta 專屬特徵工程。
這些特徵在一般股票模型裡不存在，
但對 IPO 後前 90 天的預測力最強。

Usage
-----
from mtf_modules.ipo_features import build_ipo_features
feat = build_ipo_features(df_1m, df_daily, profile)
"""

import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")


# ──────────────────────────────────────────────
# ANCHORED VWAP（IPO 日錨定）
# ──────────────────────────────────────────────

def anchored_vwap(df: pd.DataFrame,
                  anchor_date: str | pd.Timestamp) -> pd.Series:
    """
    從 IPO 日起累計的 VWAP。
    機構平均持倉成本線，是最關鍵的支撐壓力位。
    """
    anchor = pd.Timestamp(anchor_date)
    if df.index.tzinfo is not None and anchor.tzinfo is None:
        anchor = ET.localize(anchor)
    sub = df[df.index >= anchor].copy()
    if sub.empty:
        return pd.Series(np.nan, index=df.index, name="avwap_ipo")
    tp      = (sub["high"] + sub["low"] + sub["close"]) / 3
    cum_tpv = (tp * sub["volume"]).cumsum()
    cum_vol = sub["volume"].cumsum()
    avwap   = (cum_tpv / (cum_vol + 1e-9)).rename("avwap_ipo")
    return avwap.reindex(df.index)


# ──────────────────────────────────────────────
# 盤前訊號
# ──────────────────────────────────────────────

def premarket_signals(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    盤前（04:00–09:30 ET）量價訊號。
    對 IPO 高 Beta 股，盤前量是隔日最強預測因子之一。
    """
    df = df_1m.copy()
    df["_date"] = df.index.normalize()
    is_pm = df.index.time < pd.Timestamp("09:30").time()
    is_reg= ((df.index.time >= pd.Timestamp("09:30").time()) &
             (df.index.time <= pd.Timestamp("16:00").time()))

    pm_df  = df[is_pm]
    reg_df = df[is_reg]

    # 盤前日匯總
    pm_vol   = pm_df.groupby("_date")["volume"].sum().rename("pm_vol")
    pm_open  = pm_df.groupby("_date")["open"].first()
    pm_close = pm_df.groupby("_date")["close"].last()
    pm_high  = pm_df.groupby("_date")["high"].max()
    pm_low   = pm_df.groupby("_date")["low"].min()
    pm_ret   = ((pm_close - pm_open) / (pm_open + 1e-9)).rename("pm_ret")

    # 前日正常盤量
    prev_reg_vol = reg_df.groupby("_date")["volume"].sum().shift(1).rename("prev_reg_vol")
    prev_close   = reg_df.groupby("_date")["close"].last().shift(1).rename("prev_close")

    # 盤前量比
    pm_ratio = (pm_vol / (prev_reg_vol + 1e-9)).rename("pm_vol_ratio")

    # 隔夜 Gap
    gap = ((pm_open - prev_close) / (prev_close + 1e-9)).rename("overnight_gap")

    # 盤前活躍度分級
    pm_active  = (pm_ratio > 0.15).astype(int).rename("pm_active")
    pm_extreme = (pm_ratio > 0.40).astype(int).rename("pm_extreme")

    # 盤前方向
    pm_bull    = (pm_ret > 0.005).astype(int).rename("pm_bull")
    pm_bear    = (pm_ret < -0.005).astype(int).rename("pm_bear")

    # 組合成日級 DataFrame
    daily_feat = pd.concat([
        pm_vol, pm_ret, gap,
        pm_ratio, pm_active, pm_extreme,
        pm_bull, pm_bear,
    ], axis=1)

    # 廣播到 1m index（每根 bar 知道當日的盤前狀況）
    daily_feat.index = pd.to_datetime(daily_feat.index)
    result = daily_feat.reindex(
        df_1m.index.normalize().unique()
    ).reindex(df_1m.index.normalize(), method="ffill")
    result.index = df_1m.index
    return result.fillna(0)


# ──────────────────────────────────────────────
# 收盤統計（當日收盤後用於隔日預測）
# ──────────────────────────────────────────────

def daily_close_features(df_daily: pd.DataFrame,
                          profile: dict) -> pd.DataFrame:
    """
    每日收盤後產生的特徵，用於預測隔日方向。
    這是跨日預測的核心特徵集。
    """
    f = pd.DataFrame(index=df_daily.index)
    c = df_daily["close"]
    o = df_daily["open"]
    h = df_daily["high"]
    l = df_daily["low"]
    v = df_daily["volume"]

    # ── 1. 收盤位置（最重要）──
    # 收在當日高點附近 → 隔日偏多
    f["close_location"] = (c - l) / (h - l + 1e-9)
    f["close_near_high"] = (f["close_location"] > 0.8).astype(int)
    f["close_near_low"]  = (f["close_location"] < 0.2).astype(int)

    # ── 2. 當日報酬率 ──
    f["day_ret"]     = c.pct_change()
    f["day_ret_3d"]  = c.pct_change(3)
    f["day_ret_5d"]  = c.pct_change(5)
    f["day_up"]      = (f["day_ret"] > 0).astype(int)

    # ── 3. 相對 IPO 價格 ──
    ipo_price = float(profile.get("ipo_price") or
                      df_daily["open"].iloc[0])
    f["vs_ipo_price"]  = (c - ipo_price) / ipo_price
    f["above_ipo"]     = (c > ipo_price).astype(int)

    # ── 4. 振幅（高振幅後可能縮量整理）──
    f["day_range"]    = (h - l) / (c + 1e-9)
    f["day_range_ma"] = f["day_range"].rolling(5, min_periods=1).mean()
    f["range_ratio"]  = f["day_range"] / (f["day_range_ma"] + 1e-9)

    # ── 5. 量能 ──
    vol_ma5 = v.rolling(5, min_periods=1).mean()
    f["rvol_day"]    = v / (vol_ma5 + 1e-9)
    f["vol_up"]      = (f["rvol_day"] > 1.5).astype(int)
    f["vol_extreme"] = (f["rvol_day"] > 3.0).astype(int)

    # ── 6. Float Utilization（日線級）──
    float_shares = profile.get("float_shares")
    if float_shares and float_shares > 0:
        f["float_util_day"]    = v / float_shares
        f["float_extreme_day"] = (f["float_util_day"] > 0.3).astype(int)
    else:
        f["float_util_day"]    = 0.0
        f["float_extreme_day"] = 0

    # ── 7. 當日 VWAP 位置 ──
    tp   = (h + l + c) / 3
    vwap = (tp * v).cumsum() / (v.cumsum() + 1e-9)
    f["vs_vwap_day"]     = (c - vwap) / (vwap + 1e-9)
    f["above_vwap_day"]  = (c > vwap).astype(int)

    # ── 8. 動能連續性 ──
    f["consec_up"]  = 0
    f["consec_down"]= 0
    up_streak = dn_streak = 0
    for i, r in enumerate(f["day_ret"].fillna(0)):
        if r > 0:
            up_streak += 1; dn_streak = 0
        elif r < 0:
            dn_streak += 1; up_streak = 0
        f.iloc[i, f.columns.get_loc("consec_up")]   = up_streak
        f.iloc[i, f.columns.get_loc("consec_down")] = dn_streak

    # ── 9. IPO 天數（新鮮度）──
    ipo_date = profile.get("ipo_date")
    if ipo_date:
        ipo_ts = pd.Timestamp(ipo_date)

        # ── 修正：統一時區 ──
        # df_daily.index 可能是 tz-aware（ET），ipo_ts 是 tz-naive
        # 統一轉為 tz-naive date 計算天數
        idx_dates = df_daily.index.normalize()
        if idx_dates.tzinfo is not None:
            idx_dates = idx_dates.tz_localize(None)

        ipo_ts_naive = ipo_ts.tz_localize(None) if ipo_ts.tzinfo else ipo_ts

        f["ipo_days"]    = np.maximum((idx_dates - ipo_ts_naive).days.values, 0)
        f["ipo_week1"]   = (f["ipo_days"] <= 5).astype(int)
        f["ipo_week2"]   = ((f["ipo_days"] > 5) &
                            (f["ipo_days"] <= 10)).astype(int)
        f["lockup_fear"] = ((90 - f["ipo_days"]).clip(lower=0) < 14).astype(int)
    else:
        f["ipo_days"]    = 30
        f["ipo_week1"]   = 0
        f["ipo_week2"]   = 0
        f["lockup_fear"] = 0

    # ── 10. 技術指標（日線）──
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(5, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(5, min_periods=1).mean()
    f["rsi_daily"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # EMA cross
    ema5  = c.ewm(span=5,  adjust=False).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    f["ema_cross_daily"] = (ema5 - ema10) / (c + 1e-9)
    f["ema_sign_daily"]  = np.sign(f["ema_cross_daily"])

    # BB position
    ma10 = c.rolling(10, min_periods=1).mean()
    std10= c.rolling(10, min_periods=1).std().fillna(0)
    bb_lo= ma10 - 2*std10; bb_hi = ma10 + 2*std10
    f["bb_pos_daily"]   = (c - bb_lo) / (bb_hi - bb_lo + 1e-9)
    f["bb_width_daily"] = (bb_hi - bb_lo) / (ma10 + 1e-9)

    return f


# ──────────────────────────────────────────────
# 新聞情緒（免費 yfinance）
# ──────────────────────────────────────────────

def news_sentiment(ticker: str) -> dict:
    """
    用 yfinance.news + 關鍵字打分。
    不需要任何付費 API。
    """
    import yfinance as yf

    POSITIVE = [
        "upgrade", "buy", "strong buy", "beat", "surge", "soar", "rally",
        "bullish", "outperform", "raised", "record", "growth", "beat",
        # IPO 特有正面觸發詞
        "fast track", "index inclusion", "sp500", "s&p 500",
        "added to", "joins index", "etf added",
        "partnership", "contract", "deal", "wins",
    ]
    NEGATIVE = [
        "downgrade", "sell", "underperform", "miss", "crash", "plunge",
        "bearish", "cut", "warning", "concern", "investigation",
        # IPO 特有負面觸發詞
        "lockup expir", "insider selling", "dilut", "secondary offering",
        "short seller", "fraud", "sec inquiry",
    ]
    STRONG_POSITIVE = ["fast track", "index inclusion", "sp500", "added to index"]
    STRONG_NEGATIVE = ["lockup expir", "short seller report", "sec inquiry"]

    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []

    scores     = []
    titles     = []
    has_strong_pos = False
    has_strong_neg = False

    for item in news[:30]:
        title   = (item.get("title")   or "").lower()
        summary = (item.get("summary") or "").lower()
        text    = title + " " + summary
        titles.append(item.get("title", ""))

        # 強訊號偵測
        if any(kw in text for kw in STRONG_POSITIVE):
            has_strong_pos = True
        if any(kw in text for kw in STRONG_NEGATIVE):
            has_strong_neg = True

        pos = sum(1 for kw in POSITIVE  if kw in text)
        neg = sum(1 for kw in NEGATIVE  if kw in text)
        scores.append(pos - neg)

    total_score = sum(scores)
    avg_score   = np.mean(scores) if scores else 0

    return {
        "news_count"      : len(news),
        "sentiment_score" : total_score,
        "sentiment_avg"   : round(avg_score, 3),
        "sentiment_signal": ("POSITIVE" if total_score > 2
                              else "NEGATIVE" if total_score < -2
                              else "NEUTRAL"),
        "has_index_news"  : has_strong_pos,
        "has_lockup_news" : has_strong_neg,
        "recent_titles"   : titles[:5],
    }


# ──────────────────────────────────────────────
# 主組裝函數
# ──────────────────────────────────────────────

def build_ipo_features(df_1m:    pd.DataFrame,
                        df_daily: pd.DataFrame,
                        profile:  dict,
                        sentiment: dict | None = None) -> pd.DataFrame:
    """
    組裝完整的 IPO 高 Beta 特徵矩陣（日線級，用於隔日預測）。

    Parameters
    ----------
    df_1m     : 1m OHLCV（用於盤前訊號提取）
    df_daily  : 日線 OHLCV
    profile   : dict from scanner.fetch_ticker_profile()
    sentiment : dict from news_sentiment()（可選）

    Returns
    -------
    features : DataFrame，index = df_daily.index
               每行 = 某日收盤後的特徵，用於預測隔日方向
    """
    # 日線特徵
    feat = daily_close_features(df_daily, profile)

    # ── 統一輔助函數：去除時區後 reindex ──
    def _reindex_to_daily(series: pd.Series) -> pd.Series:
        """把 tz-aware series 對齊到 df_daily.index（去除時區後比對）。"""
        # df_daily.index 轉為 tz-naive 用於 reindex key
        daily_idx_naive = df_daily.index.tz_localize(None) \
                          if df_daily.index.tzinfo is not None \
                          else df_daily.index
        series_naive = series.copy()
        if series.index.tzinfo is not None:
            series_naive.index = series_naive.index.tz_localize(None)
        return series_naive.reindex(daily_idx_naive, method="ffill")

    # AVWAP（IPO 日錨定）
    if profile.get("ipo_date") and not df_1m.empty:
        avwap = anchored_vwap(df_1m, str(profile["ipo_date"]))
        # resample to daily, then align — strip tz before reindex
        avwap_rs = avwap.resample("1D").last()
        avwap_daily = _reindex_to_daily(avwap_rs)
        feat["avwap_ipo"]       = avwap_daily.values
        feat["vs_avwap_ipo"]    = (df_daily["close"].values -
                                    avwap_daily.values) / (avwap_daily.values + 1e-9)
        feat["above_avwap_ipo"] = (df_daily["close"].values >
                                    avwap_daily.values).astype(int)

    # 盤前訊號（對齊到日線）
    if not df_1m.empty:
        pm = premarket_signals(df_1m)
        pm_daily = pm.resample("1D").last()
        for col in pm_daily.columns:
            aligned = _reindex_to_daily(pm_daily[col])
            feat[col] = aligned.values

    # 新聞情緒
    if sentiment:
        feat["news_count"]      = sentiment.get("news_count", 0)
        feat["sentiment_score"] = sentiment.get("sentiment_score", 0)
        feat["has_index_news"]  = int(sentiment.get("has_index_news", False))
        feat["has_lockup_news"] = int(sentiment.get("has_lockup_news", False))
        sig = sentiment.get("sentiment_signal", "NEUTRAL")
        feat["sentiment_pos"]   = int(sig == "POSITIVE")
        feat["sentiment_neg"]   = int(sig == "NEGATIVE")

    # SPY 相對強度（日線）— tz-aware reindex 修正
    try:
        import yfinance as yf
        spy = yf.download(
            "SPY",
            start=df_daily.index[0].strftime("%Y-%m-%d"),
            end=(df_daily.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d", progress=False,
            auto_adjust=True, multi_level_index=False,
        )
        if not spy.empty:
            spy.columns = [c.lower() for c in spy.columns]
            # 統一去除時區
            spy.index = pd.to_datetime(spy.index)
            if spy.index.tzinfo is not None:
                spy.index = spy.index.tz_localize(None)

            daily_idx_naive = df_daily.index.tz_localize(None) \
                              if df_daily.index.tzinfo is not None \
                              else df_daily.index

            spy_ret    = spy["close"].pct_change()
            tkr_close  = df_daily["close"].copy()
            tkr_close.index = daily_idx_naive
            tkr_ret    = tkr_close.pct_change()
            spy_ret_al = spy_ret.reindex(daily_idx_naive, method="ffill").fillna(0)

            rs = (tkr_ret - spy_ret_al).rolling(3, min_periods=1).mean()
            feat["rs_spy_daily"] = rs.values
            feat["rs_spy_sign"]  = np.sign(rs).values
    except Exception:
        feat["rs_spy_daily"] = 0.0
        feat["rs_spy_sign"]  = 0.0

    return feat
