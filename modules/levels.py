"""
mtf_modules/levels.py  v2
────────────────────────────────────────────
完整技術位進出點位計算引擎。

技術位來源（優先序）：
  1. Pivot Point + R1/R2/S1/S2
  2. Opening Range High/Low
  3. 近期 Swing High/Low（5/10/20日）
  4. EMA 5/10/20（動態支撐壓力）
  5. VWAP（日線當日）
  6. Anchored VWAP（IPO 日）
  7. ATR 倍數作為間距補充

多空皆支援。強訊號才輸出點位。

Usage
-----
from mtf_modules.levels import calculate_levels, format_levels, LevelEngine
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class PriceLevel:
    price:  float
    label:  str
    source: str          # pivot / swing / ema / vwap / avwap / atr
    weight: float = 1.0  # 越高越可信

    def __repr__(self):
        return f"{self.label}: ${self.price:.2f} ({self.source})"


@dataclass
class LevelSet:
    """所有技術位的容器。"""
    supports:    list[PriceLevel] = field(default_factory=list)
    resistances: list[PriceLevel] = field(default_factory=list)
    last_close:  float = 0.0
    atr:         float = 0.0

    def best_support(self, n: int = 1) -> list[PriceLevel]:
        """最近的 n 個支撐位（排除過近的重複）"""
        return _deduplicate(
            sorted(self.supports, key=lambda x: x.price, reverse=True),
            self.atr * 0.3, n
        )

    def best_resistance(self, n: int = 1) -> list[PriceLevel]:
        """最近的 n 個壓力位"""
        return _deduplicate(
            sorted(self.resistances, key=lambda x: x.price),
            self.atr * 0.3, n
        )


def _deduplicate(levels: list[PriceLevel],
                 min_gap: float,
                 n: int) -> list[PriceLevel]:
    """移除過於接近的重複位，保留最高權重的。"""
    result = []
    for lv in levels:
        if not result or abs(lv.price - result[-1].price) >= min_gap:
            result.append(lv)
        elif lv.weight > result[-1].weight:
            result[-1] = lv
        if len(result) >= n:
            break
    return result


# ──────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return float(df["close"].iloc[-1]) * 0.03
    c = df["close"]; h = df["high"]; l = df["low"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=1).mean().iloc[-1])


# ──────────────────────────────────────────────
# LEVEL ENGINE
# ──────────────────────────────────────────────

class LevelEngine:
    """
    從日線 OHLCV 萃取全套技術支撐壓力位。
    每種方法獨立計算，最後統一分類為支撐/壓力。
    """

    def __init__(self, df: pd.DataFrame,
                 profile: dict | None = None,
                 avwap_val: float | None = None):
        """
        Parameters
        ----------
        df        : 日線 OHLCV，index = DatetimeIndex
        profile   : ticker profile dict（提供 ipo_date, float_shares 等）
        avwap_val : IPO 日錨定 VWAP 當前值（可選）
        """
        self.df       = df.copy()
        self.profile  = profile or {}
        self.avwap    = avwap_val
        self.close    = float(df["close"].iloc[-1])
        self.atr_val  = _atr(df)
        self._levels: list[PriceLevel] = []
        self._compute_all()

    # ── 各技術位計算 ──

    def _add(self, price: float, label: str,
             source: str, weight: float = 1.0):
        """加入一個技術位（過濾 NaN 和負值）。"""
        if price and not np.isnan(price) and price > 0:
            self._levels.append(PriceLevel(
                price=round(float(price), 4),
                label=label, source=source, weight=weight,
            ))

    def _pivot_points(self):
        """Classic Pivot Point（前日高低收）。"""
        if len(self.df) < 2:
            return
        prev = self.df.iloc[-2]
        H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        R2 = PP + (H - L)
        R3 = H + 2 * (PP - L)
        S1 = 2 * PP - H
        S2 = PP - (H - L)
        S3 = L - 2 * (H - PP)

        self._add(PP, "Pivot",  "pivot", weight=2.0)
        self._add(R1, "R1",     "pivot", weight=2.5)
        self._add(R2, "R2",     "pivot", weight=2.0)
        self._add(R3, "R3",     "pivot", weight=1.5)
        self._add(S1, "S1",     "pivot", weight=2.5)
        self._add(S2, "S2",     "pivot", weight=2.0)
        self._add(S3, "S3",     "pivot", weight=1.5)

    def _swing_highs_lows(self):
        """近期 Swing High/Low（5/10/20 日）。"""
        h = self.df["high"]
        l = self.df["low"]
        for n, w in [(5, 2.0), (10, 1.8), (20, 1.5)]:
            if len(self.df) >= n:
                self._add(float(h.rolling(n).max().iloc[-1]),
                          f"{n}d High", "swing", weight=w)
                self._add(float(l.rolling(n).min().iloc[-1]),
                          f"{n}d Low",  "swing", weight=w)

        # 前日高低
        if len(self.df) >= 2:
            self._add(float(h.iloc[-2]), "Prev Day High",
                      "swing", weight=2.2)
            self._add(float(l.iloc[-2]), "Prev Day Low",
                      "swing", weight=2.2)

    def _ema_levels(self):
        """EMA 5/10/20/50（動態支撐壓力）。"""
        c = self.df["close"]
        for span, w in [(5, 1.8), (10, 1.6), (20, 1.5), (50, 1.3)]:
            if len(self.df) >= span:
                val = float(c.ewm(span=span, adjust=False).mean().iloc[-1])
                self._add(val, f"EMA{span}", "ema", weight=w)

    def _vwap_daily(self):
        """當日 VWAP（最後一天）。"""
        last = self.df.iloc[-1]
        tp   = (float(last["high"]) + float(last["low"]) +
                float(last["close"])) / 3
        self._add(tp, "VWAP Today", "vwap", weight=2.0)

    def _avwap_ipo(self):
        """IPO 日錨定 VWAP。"""
        if self.avwap:
            self._add(self.avwap, "AVWAP (IPO)",
                      "avwap", weight=2.8)   # 最高權重

    def _opening_range(self):
        """前日 Opening Range（日線近似：前日開盤後高低）。"""
        if len(self.df) >= 2:
            prev = self.df.iloc[-2]
            self._add(float(prev["high"]), "Prev OR High",
                      "or", weight=1.9)
            self._add(float(prev["low"]),  "Prev OR Low",
                      "or", weight=1.9)

    def _compute_all(self):
        self._pivot_points()
        self._swing_highs_lows()
        self._ema_levels()
        self._vwap_daily()
        self._avwap_ipo()
        self._opening_range()

    def get_levelset(self) -> LevelSet:
        """分類所有技術位為支撐/壓力。"""
        ls = LevelSet(last_close=self.close, atr=self.atr_val)
        buf = self.atr_val * 0.15   # 緩衝：避免與收盤太近的位進入

        for lv in self._levels:
            if lv.price < self.close - buf:
                ls.supports.append(lv)
            elif lv.price > self.close + buf:
                ls.resistances.append(lv)
        return ls

    def all_levels_df(self) -> pd.DataFrame:
        """所有技術位 DataFrame（供調試/顯示）。"""
        rows = []
        for lv in self._levels:
            rows.append({
                "label"    : lv.label,
                "price"    : lv.price,
                "source"   : lv.source,
                "weight"   : lv.weight,
                "vs_close" : round(lv.price / self.close - 1, 4),
                "side"     : "support" if lv.price < self.close
                              else "resistance",
            })
        return (pd.DataFrame(rows)
                .sort_values("price", ascending=False)
                .reset_index(drop=True))


# ──────────────────────────────────────────────
# MAIN CALCULATE FUNCTION
# ──────────────────────────────────────────────

def calculate_levels(
    df_daily:     pd.DataFrame,
    profile:      dict,
    direction:    str,
    proba:        np.ndarray,
    atr_mult_sl:  float = 1.5,
    atr_mult_tp1: float = 1.5,
    atr_mult_tp2: float = 3.0,
    atr_mult_tp3: float = 5.0,
    avwap_val:    float | None = None,
) -> dict | None:
    """
    計算多空進出點位。

    Strategy
    --------
    - 進場：理想 = 收盤；積極 = 回測最近支撐/壓力附近；
              保守 = 突破確認（+ATR×0.3）
    - 止損：最近技術位 vs ATR 倍數 → 取對交易者較嚴格者
    - 目標：技術位加權融合 + ATR 備用
              T1 = 最近技術位
              T2 = 次近技術位 or ATR×tp2
              T3 = ATR×tp3（高 Beta 延伸）
    - Float 調整：換手 > 50% → 止損擴大 25%

    Returns None if direction is NEUTRAL or UNCERTAIN.
    """
    if direction in ("NEUTRAL", "UNCERTAIN"):
        return None
    if df_daily.empty:
        return None

    # ── 建構技術位引擎 ──
    engine   = LevelEngine(df_daily, profile, avwap_val)
    ls       = engine.get_levelset()
    atr_val  = engine.atr_val
    last_close = engine.close

    last_high = float(df_daily["high"].iloc[-1])
    last_low  = float(df_daily["low"].iloc[-1])

    is_long   = direction in ("STRONG_UP", "WEAK_UP")
    is_strong = direction in ("STRONG_UP", "STRONG_DOWN")
    confidence = float(max(proba))

    # Float 換手調整
    float_shares = profile.get("float_shares")
    float_util   = None
    float_mult   = 1.0
    if float_shares and float_shares > 0:
        last_vol   = float(df_daily["volume"].iloc[-1])
        float_util = last_vol / float_shares
        if float_util > 0.5:
            float_mult = 1.25   # 擴大止損/目標

    # ── 多頭 ──
    if is_long:
        # 進場
        entry_ideal        = round(last_close, 2)
        entry_aggressive   = round(
            ls.best_support(1)[0].price * 0.998
            if ls.best_support(1) else last_close * 0.995, 2
        )
        entry_conservative = round(last_close + atr_val * 0.3, 2)

        # 止損
        atr_sl = last_close - atr_val * atr_mult_sl * float_mult
        tech_sl = (ls.best_support(1)[0].price - atr_val * 0.2
                   if ls.best_support(1) else atr_sl)
        stop_loss = round(max(atr_sl, tech_sl), 2)   # 較嚴格（較高）

        # 目標
        res = ls.best_resistance(3)
        t1_tech = res[0].price if len(res) >= 1 else None
        t2_tech = res[1].price if len(res) >= 2 else None
        t3_tech = res[2].price if len(res) >= 3 else None

        t1_atr = last_close + atr_val * atr_mult_tp1 * float_mult
        t2_atr = last_close + atr_val * atr_mult_tp2 * float_mult
        t3_atr = last_close + atr_val * atr_mult_tp3

        target_1 = round((t1_tech + t1_atr) / 2 if t1_tech else t1_atr, 2)
        target_2 = round((t2_tech + t2_atr) / 2 if t2_tech else t2_atr, 2)
        target_3 = round(t3_tech if t3_tech else t3_atr, 2)

        # 目標一定要遞增
        target_1 = max(target_1, round(last_close + atr_val * 0.8, 2))
        target_2 = max(target_2, round(target_1 + atr_val * 1.0, 2))
        target_3 = max(target_3, round(target_2 + atr_val * 1.5, 2))

        invalidation = round(last_low - atr_val * 0.5, 2)

        # 止損標籤
        sl_source = (ls.best_support(1)[0].label
                     if ls.best_support(1) else "ATR")
        t1_source = res[0].label if len(res) >= 1 else "ATR"
        t2_source = res[1].label if len(res) >= 2 else "ATR×3"
        t3_source = res[2].label if len(res) >= 3 else "ATR×5"

    # ── 空頭 ──
    else:
        entry_ideal        = round(last_close, 2)
        entry_aggressive   = round(
            ls.best_resistance(1)[0].price * 1.002
            if ls.best_resistance(1) else last_close * 1.005, 2
        )
        entry_conservative = round(last_close - atr_val * 0.3, 2)

        atr_sl  = last_close + atr_val * atr_mult_sl * float_mult
        tech_sl = (ls.best_resistance(1)[0].price + atr_val * 0.2
                   if ls.best_resistance(1) else atr_sl)
        stop_loss = round(min(atr_sl, tech_sl), 2)   # 較嚴格（較低）

        sup = ls.best_support(3)
        t1_tech = sup[0].price if len(sup) >= 1 else None
        t2_tech = sup[1].price if len(sup) >= 2 else None
        t3_tech = sup[2].price if len(sup) >= 3 else None

        t1_atr = last_close - atr_val * atr_mult_tp1 * float_mult
        t2_atr = last_close - atr_val * atr_mult_tp2 * float_mult
        t3_atr = last_close - atr_val * atr_mult_tp3

        target_1 = round((t1_tech + t1_atr) / 2 if t1_tech else t1_atr, 2)
        target_2 = round((t2_tech + t2_atr) / 2 if t2_tech else t2_atr, 2)
        target_3 = round(t3_tech if t3_tech else t3_atr, 2)

        # 目標一定要遞減
        target_1 = min(target_1, round(last_close - atr_val * 0.8, 2))
        target_2 = min(target_2, round(target_1 - atr_val * 1.0, 2))
        target_3 = min(target_3, round(target_2 - atr_val * 1.5, 2))

        invalidation = round(last_high + atr_val * 0.5, 2)

        sl_source = (ls.best_resistance(1)[0].label
                     if ls.best_resistance(1) else "ATR")
        t1_source = sup[0].label if len(sup) >= 1 else "ATR"
        t2_source = sup[1].label if len(sup) >= 2 else "ATR×3"
        t3_source = sup[2].label if len(sup) >= 3 else "ATR×5"

    # ── Risk/Reward ──
    risk = abs(entry_ideal - stop_loss)
    rr1  = round(abs(target_1 - entry_ideal) / (risk + 1e-9), 2)
    rr2  = round(abs(target_2 - entry_ideal) / (risk + 1e-9), 2)
    rr3  = round(abs(target_3 - entry_ideal) / (risk + 1e-9), 2)

    # ── 部位備注 ──
    if is_strong and confidence >= 0.60:
        pos_note = "✅ 強訊號 — 標準部位（建議不超過帳戶 5%）"
    elif is_strong:
        pos_note = "⚡ 中強訊號 — 建議半倉，確認後加倉"
    else:
        pos_note = "⚠️ 弱訊號 — 輕倉試探（1/4 倉），嚴守止損"

    ipo_days = profile.get("ipo_days", 99)
    if ipo_days <= 5:
        pos_note += "\n🔥 IPO 第一週，波動極大，部位再減半"
    elif ipo_days <= 10:
        pos_note += "\n⚡ IPO 第二週，仍有高波動風險"

    if float_util and float_util > 0.5:
        pos_note += f"\n⚠️ 昨日 Float 換手 {float_util:.0%}，止損已擴大"

    return {
        # 方向
        "direction"           : direction,
        "is_long"             : is_long,
        "is_strong"           : is_strong,
        "confidence"          : round(confidence, 4),

        # 價格參考
        "last_close"          : last_close,
        "atr"                 : round(atr_val, 4),
        "float_util"          : round(float_util, 4) if float_util else None,

        # 進場
        "entry_ideal"         : entry_ideal,
        "entry_aggressive"    : entry_aggressive,
        "entry_conservative"  : entry_conservative,

        # 出場
        "stop_loss"           : stop_loss,
        "stop_loss_source"    : sl_source,
        "target_1"            : target_1,
        "target_1_source"     : t1_source,
        "target_2"            : target_2,
        "target_2_source"     : t2_source,
        "target_3"            : target_3,
        "target_3_source"     : t3_source,
        "invalidation_level"  : invalidation,

        # 風報比
        "risk_per_share"      : round(risk, 4),
        "risk_reward_1"       : rr1,
        "risk_reward_2"       : rr2,
        "risk_reward_3"       : rr3,

        # 備注
        "position_note"       : pos_note,

        # 完整技術位（供 debug / 圖表用）
        "level_df"            : engine.all_levels_df(),
        "levelset"            : ls,
    }


# ──────────────────────────────────────────────
# FORMATTER
# ──────────────────────────────────────────────

def format_levels(lv: dict) -> str:
    if lv is None:
        return "_方向為 NEUTRAL / UNCERTAIN，不提供點位。_"

    arrow = "↑" if lv["is_long"] else "↓"
    side  = "📈 做多" if lv["is_long"] else "📉 做空"

    return "\n".join([
        f"**{side}**",
        "",
        "| 項目 | 價格 | 來源 | 說明 |",
        "|------|------|------|------|",
        f"| 🎯 理想進場 | **${lv['entry_ideal']}** | 前日收盤 | 開盤附近進場 |",
        f"| ⚡ 積極進場 | ${lv['entry_aggressive']} | 技術位 | 回測支撐/壓力 |",
        f"| 🛡️ 保守進場 | ${lv['entry_conservative']} | +ATR | 確認方向後 |",
        f"| 🔴 止損 | **${lv['stop_loss']}** | {lv['stop_loss_source']} | 跌破/突破即出 |",
        f"| {arrow} 目標一 | **${lv['target_1']}** | {lv['target_1_source']} | RR {lv['risk_reward_1']}× |",
        f"| {arrow} 目標二 | ${lv['target_2']} | {lv['target_2_source']} | RR {lv['risk_reward_2']}× |",
        f"| {arrow} 目標三 | ${lv['target_3']} | {lv['target_3_source']} | RR {lv['risk_reward_3']}× |",
        f"| ❌ 訊號失效 | ${lv['invalidation_level']} | 前日高/低 | 超過放棄訊號 |",
        "",
        f"**每股風險：** ${lv['risk_per_share']} &nbsp;|&nbsp; **ATR：** ${lv['atr']}",
        "",
        f"_{lv['position_note']}_",
    ])
