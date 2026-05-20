"""
mtf_modules/labels.py
────────────────────────────────────────────
Label generation for supervised learning.

Usage
-----
from mtf_modules.labels import make_labels, dynamic_confidence_gate
"""

import numpy as np
import pandas as pd


def make_labels(df: pd.DataFrame,
                horizon:  int   = 10,
                bt:       float = 0.006,
                ct:       float = -0.006,
                use_atr:  bool  = True,
                atr_mult: float = 1.0) -> pd.Series:
    """
    Forward-looking 3-class label.

    Classes
    -------
    2 = BREAKOUT  : future max return  > threshold
    0 = CRASH     : future min return  < threshold
    1 = SQUEEZE   : everything else

    Parameters
    ----------
    horizon  : number of forward bars to look ahead
    bt / ct  : fixed breakout / crash threshold (used if use_atr=False)
    use_atr  : if True, scale thresholds by ATR/close (better for high-beta)
    atr_mult : ATR multiplier for dynamic threshold
    """
    c  = df["close"]
    fh = c.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    fl = c.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    fm = (fh - c) / (c + 1e-9)
    fn = (fl - c) / (c + 1e-9)

    if use_atr and "m1_atr14" in df.columns:
        atr_norm = (df["m1_atr14"].rolling(14, min_periods=1).mean()
                    / (c + 1e-9) * atr_mult)
        bt_dyn = atr_norm.clip(lower=abs(bt))
        ct_dyn = -atr_norm.clip(lower=abs(ct))
    else:
        bt_dyn = pd.Series(bt,  index=df.index)
        ct_dyn = pd.Series(ct,  index=df.index)

    lb = pd.Series(1, index=df.index, name="label")
    lb[fm >  bt_dyn] = 2
    lb[fn <  ct_dyn] = 0

    # Tie-break: both triggered → take stronger signal
    both = (fm > bt_dyn) & (fn < ct_dyn)
    lb[both & (fm.abs() >= fn.abs())] = 2
    lb[both & (fm.abs() <  fn.abs())] = 0

    return lb


def dynamic_confidence_gate(base_gate:  float,
                             vol_regime: float | None) -> float:
    """
    Adjust confidence gate based on volatility regime.

    Regime 0 (low vol)   → gate - 5%  (easier threshold, signal cleaner)
    Regime 1 (mid vol)   → gate unchanged
    Regime 2 (high burst)→ gate + 10% (harder threshold, fewer false signals)
    """
    if vol_regime is None:
        return base_gate
    if   vol_regime >= 2: return min(base_gate + 0.10, 0.75)
    elif vol_regime >= 1: return base_gate
    else:                 return max(base_gate - 0.05, 0.30)


def label_summary(y: pd.Series) -> dict:
    """Return count + pct of each class."""
    total = len(y)
    vc    = y.value_counts()
    return {
        "crash"   : int(vc.get(0, 0)),
        "squeeze" : int(vc.get(1, 0)),
        "breakout": int(vc.get(2, 0)),
        "total"   : total,
        "crash_pct"   : round(vc.get(0, 0) / total * 100, 1),
        "squeeze_pct" : round(vc.get(1, 0) / total * 100, 1),
        "breakout_pct": round(vc.get(2, 0) / total * 100, 1),
    }
