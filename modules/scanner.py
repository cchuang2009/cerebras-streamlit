"""
mtf_modules/scanner.py
────────────────────────────────────────────
自動掃描近期 IPO 高 Beta 標的。
完全免費：yfinance + 靜態 IPO 清單 + 自動篩選條件。

Usage
-----
from mtf_modules.scanner import scan_ipo_candidates
candidates = scan_ipo_candidates(max_ipo_days=60, min_beta=3.0)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 靜態 IPO 種子清單（近期高知名度 IPO）
# 使用者可自行增刪
# ──────────────────────────────────────────────

KNOWN_IPO_TICKERS = [
    # AI / 半導體
    "CBRS",   # Cerebras Systems   IPO 2026-05-14
    "CRWV",   # CoreWeave          IPO 2025-03
    "NBIS",   # Nebius Group       IPO 2024
    # 光通訊 / AI 基建
    "AAOI",   # Applied Optoelectronics（高 Beta 老將）
    "ALAB",   # Astera Labs        IPO 2024
    # 其他近期高 Beta
    "RDDT",   # Reddit             IPO 2024
    "ASTERA", # placeholder
]

# 移除無效 ticker
KNOWN_IPO_TICKERS = [t for t in KNOWN_IPO_TICKERS if t != "ASTERA"]


# ──────────────────────────────────────────────
# 篩選條件
# ──────────────────────────────────────────────

IPO_SCAN_CRITERIA = {
    "max_ipo_days"     : 90,    # IPO 後幾天內
    "min_beta"         : 3.0,   # 最低 Beta
    "min_avg_volume"   : 1_000_000,   # 最低日均量
    "max_market_cap"   : 200_000_000_000,  # 最大市值（太大的不夠 volatile）
    "min_price"        : 5.0,   # 避免仙股
}


def _safe_get(info: dict, key: str, default=None):
    val = info.get(key)
    return val if val is not None else default


def fetch_ticker_profile(ticker: str) -> dict | None:
    """
    拉取單一 ticker 的基本資料。
    Returns None if ticker is invalid.
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        if not info or info.get("quoteType") not in ("EQUITY", "equity"):
            # 嘗試不同 quoteType key
            if not info:
                return None

        hist = tk.history(period="5d", interval="1d")
        if hist.empty:
            return None

        # IPO 日期估算：取歷史最早日期
        hist_long = tk.history(period="max", interval="1mo")
        ipo_date  = hist_long.index.min().date() if not hist_long.empty else None
        ipo_days  = (datetime.today().date() - ipo_date).days if ipo_date else 9999

        return {
            "ticker"          : ticker,
            "name"            : _safe_get(info, "longName", ticker),
            "sector"          : _safe_get(info, "sector", "Unknown"),
            "market_cap"      : _safe_get(info, "marketCap", 0),
            "beta"            : _safe_get(info, "beta", 0),
            "avg_volume"      : _safe_get(info, "averageVolume", 0),
            "avg_volume_10d"  : _safe_get(info, "averageVolume10days", 0),
            "float_shares"    : _safe_get(info, "floatShares"),
            "short_pct_float" : _safe_get(info, "shortPercentOfFloat", 0),
            "short_ratio"     : _safe_get(info, "shortRatio", 0),
            "current_price"   : _safe_get(info, "currentPrice",
                                           hist["Close"].iloc[-1] if not hist.empty else 0),
            "ipo_date"        : ipo_date,
            "ipo_days"        : ipo_days,
            "prev_close"      : hist["Close"].iloc[-1] if not hist.empty else 0,
            "prev_volume"     : hist["Volume"].iloc[-1] if not hist.empty else 0,
            "week_return"     : (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
                                  if len(hist) >= 2 else 0),
        }
    except Exception:
        return None


def score_candidate(profile: dict) -> float:
    """
    對候選標的打分（0–100）。
    分數越高 = 越適合 IPO 高 Beta 預測策略。
    """
    score = 0.0

    # Beta 分數（最重要，40 分）
    beta = profile.get("beta") or 0
    if   beta >= 10: score += 40
    elif beta >= 5 : score += 30
    elif beta >= 3 : score += 20
    elif beta >= 1 : score += 10

    # IPO 新鮮度（30 分）— 越新分越高
    days = profile.get("ipo_days", 9999)
    if   days <= 7  : score += 30
    elif days <= 30 : score += 25
    elif days <= 60 : score += 15
    elif days <= 90 : score += 8

    # 量能（20 分）
    avg_vol = profile.get("avg_volume", 0)
    if   avg_vol >= 10_000_000: score += 20
    elif avg_vol >= 5_000_000 : score += 15
    elif avg_vol >= 1_000_000 : score += 10
    elif avg_vol >= 500_000   : score += 5

    # Short Float（10 分）— 高軋空潛力
    short_pct = (profile.get("short_pct_float") or 0)
    if   short_pct >= 0.30: score += 10
    elif short_pct >= 0.15: score += 6
    elif short_pct >= 0.05: score += 3

    return round(score, 1)


def scan_ipo_candidates(
    extra_tickers: list[str] | None = None,
    max_ipo_days:  int   = 90,
    min_beta:      float = 2.0,
    min_avg_volume:int   = 500_000,
    top_n:         int   = 10,
) -> pd.DataFrame:
    """
    掃描 IPO 候選標的並排序。

    Parameters
    ----------
    extra_tickers  : 額外加入掃描的 ticker 清單
    max_ipo_days   : 只看 IPO 後 N 天內的標的
    min_beta       : 最低 Beta 門檻
    min_avg_volume : 最低日均量
    top_n          : 回傳前 N 名

    Returns
    -------
    DataFrame sorted by score descending
    """
    tickers = KNOWN_IPO_TICKERS.copy()
    if extra_tickers:
        tickers += [t.upper().strip() for t in extra_tickers]
    tickers = list(dict.fromkeys(tickers))   # deduplicate

    profiles = []
    for t in tickers:
        p = fetch_ticker_profile(t)
        if p is None:
            continue
        # 篩選條件
        if (p["beta"] or 0) < min_beta:
            continue
        if p["ipo_days"] > max_ipo_days:
            continue
        if p["avg_volume"] < min_avg_volume:
            continue
        if p["current_price"] < 5.0:
            continue
        p["score"] = score_candidate(p)
        profiles.append(p)

    if not profiles:
        return pd.DataFrame()

    df = pd.DataFrame(profiles).sort_values("score", ascending=False)
    return df.head(top_n).reset_index(drop=True)
