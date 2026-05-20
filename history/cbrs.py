"""
CBRS Multi-Timeframe Predictor — Streamlit App
pip install streamlit plotly yfinance catboost scikit-learn pytz pandas numpy prophet
Run: streamlit run cbrs_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date

import streamlit as st
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Prophet — graceful fallback if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CBRS MTF Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: Georgia, 'Times New Roman', serif;
    background-color: #f8f5f0;
    color: #2c2c2c;
}
h1, h2, h3, h4 {
    font-family: Georgia, 'Times New Roman', serif;
    color: #1a1a2e;
    font-weight: 700;
}
.main .block-container {
    background-color: #f8f5f0;
    padding-top: 2rem;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fffdf9 0%, #f3ede4 100%);
    border-right: 1px solid #ddd5c8;
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span {
    font-family: Georgia, serif;
    color: #3a3028;
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f5f0e8 100%);
    border: 1px solid #ddd5c8;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8a7968;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
    color: #1a1a2e;
}
.trend-breakout  { color: #1a7a3f; }
.trend-crash     { color: #c0392b; }
.trend-squeeze   { color: #b8860b; }
.trend-uncertain { color: #7a7060; }
.signal-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}
.badge-valid   { background:#dcf5e7; color:#1a7a3f; border:1px solid #1a7a3f; }
.badge-invalid { background:#f0ede8; color:#8a7968; border:1px solid #c8bfb4; }
hr { border-color: #ddd5c8; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stToggle"] label {
    font-family: Georgia, serif;
    color: #3a3028;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

ET = pytz.timezone("America/New_York")
CONFIDENCE_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df.sort_index()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_1m(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end,
        interval="1m", progress=False,
        auto_adjust=True, multi_level_index=False,
        prepost=True,
    )
    if df.empty:
        return pd.DataFrame()
    return _clean(df)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", progress=False,
        auto_adjust=True, multi_level_index=False,
    )
    if df.empty:
        return pd.DataFrame()
    return _clean(df)


# ─────────────────────────────────────────────
# FEATURE ENGINE
# ─────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return (df[list(agg)].resample(rule, label="left", closed="left")
            .agg(agg).dropna())


def add_indicators(df: pd.DataFrame, pfx: str) -> pd.DataFrame:
    f = df.copy()
    c, h, l, o, v = f["close"], f["high"], f["low"], f["open"], f["volume"]

    for n in [1, 3, 5, 10]:
        f[f"{pfx}_ret{n}"] = c.pct_change(n)

    if pfx != "1d":
        tp = (h + l + c) / 3
        dk = f.index.normalize()
        f[f"{pfx}_vwap"]      = (tp * v).groupby(dk).cumsum() / (v.groupby(dk).cumsum() + 1e-9)
        f[f"{pfx}_vwap_dist"] = (c - f[f"{pfx}_vwap"]) / (f[f"{pfx}_vwap"] + 1e-9)

    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    f[f"{pfx}_atr"]      = tr.rolling(14, min_periods=1).mean()
    f[f"{pfx}_atr_pct"]  = f[f"{pfx}_atr"] / (c + 1e-9)
    f[f"{pfx}_atr_ratio"]= (h - l) / (f[f"{pfx}_atr"] + 1e-9)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    f[f"{pfx}_rsi"]   = 100 - 100 / (1 + gain / (loss + 1e-9))
    f[f"{pfx}_rsi_ob"]= (f[f"{pfx}_rsi"] > 70).astype(int)
    f[f"{pfx}_rsi_os"]= (f[f"{pfx}_rsi"] < 30).astype(int)

    ma20  = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std().fillna(0)
    bb_up = ma20 + 2*std20; bb_lo = ma20 - 2*std20
    f[f"{pfx}_bb_pos"]    = (c - bb_lo) / (bb_up - bb_lo + 1e-9)
    f[f"{pfx}_bb_width"]  = (bb_up - bb_lo) / (ma20 + 1e-9)
    f[f"{pfx}_bb_squeeze"]= (f[f"{pfx}_bb_width"] < f[f"{pfx}_bb_width"].rolling(20,min_periods=1).mean()*0.8).astype(int)

    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    f[f"{pfx}_ema_cross"] = (ema9 - ema21) / (c + 1e-9)
    f[f"{pfx}_ema_sign"]  = np.sign(f[f"{pfx}_ema_cross"])
    f[f"{pfx}_ema_accel"] = f[f"{pfx}_ema_cross"].diff()

    vol_ma = v.rolling(20, min_periods=1).mean()
    f[f"{pfx}_rel_vol"]   = v / (vol_ma + 1e-9)
    f[f"{pfx}_vol_spike"] = (f[f"{pfx}_rel_vol"] > 2.5).astype(int)

    f[f"{pfx}_spread"]    = (h - l) / (c + 1e-9)
    f[f"{pfx}_imbalance"] = ((c-o)/(h-l+1e-9)).rolling(5,min_periods=1).mean()

    body = (c-o).abs(); rng = h-l+1e-9
    f[f"{pfx}_body_ratio"] = body/rng
    f[f"{pfx}_upper_wick"] = (h - pd.concat([c,o],axis=1).max(axis=1))/rng
    f[f"{pfx}_lower_wick"] = (pd.concat([c,o],axis=1).min(axis=1) - l)/rng
    f[f"{pfx}_is_bull"]    = (c > o).astype(int)
    f[f"{pfx}_volatility"] = c.pct_change().rolling(10,min_periods=1).std()
    return f


def align_to_1m(df_1m, df_htf, cols):
    shifted = df_htf[cols].shift(1)
    return shifted.reindex(df_1m.index, method="ffill")


def build_feature_matrix(df_1m: pd.DataFrame) -> pd.DataFrame:
    ind_1m  = add_indicators(df_1m,                         "m1")
    ind_5m  = add_indicators(resample_ohlcv(df_1m,"5min"),  "m5")
    ind_15m = add_indicators(resample_ohlcv(df_1m,"15min"), "m15")
    ind_30m = add_indicators(resample_ohlcv(df_1m,"30min"), "m30")

    master = ind_1m.copy()
    for ind, pfx in [(ind_5m,"m5"),(ind_15m,"m15"),(ind_30m,"m30")]:
        cols = [c for c in ind.columns if c.startswith(pfx)]
        master = pd.concat([master, align_to_1m(df_1m, ind, cols)], axis=1)

    master["time_mins"]     = np.maximum(master.index.hour*60+master.index.minute-570,0)
    master["time_sin"]      = np.sin(2*np.pi*master["time_mins"]/390)
    master["time_cos"]      = np.cos(2*np.pi*master["time_mins"]/390)
    master["is_first_30m"]  = (master["time_mins"] < 30).astype(int)
    master["is_power_hour"] = (master["time_mins"] > 330).astype(int)
    master.dropna(inplace=True)
    return master


def label_bars(df, horizon=10, bt=0.006, ct=-0.006):
    c = df["close"]
    fh = c.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fl = c.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    fm = (fh-c)/c; fn = (fl-c)/c
    lb = pd.Series(1, index=df.index)
    lb[fm > bt] = 2; lb[fn < ct] = 0
    both = (fm>bt)&(fn<ct)
    lb[both&(fm.abs()>=fn.abs())] = 2
    lb[both&(fm.abs()< fn.abs())] = 0
    return lb


def get_feat_cols(master):
    exclude = {"open","high","low","close","volume","label"}
    return [c for c in master.columns if c not in exclude]


@st.cache_resource(show_spinner=False)
def train_model(X_hash, _X, _y):
    X, y = _X, _y
    if len(X) < 50:
        return None
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return None
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sw = y_tr.map(dict(zip(classes.tolist(), weights.tolist()))).values

    model = CatBoostClassifier(
        depth=6, learning_rate=0.03, iterations=500,
        loss_function="MultiClass", eval_metric="Accuracy",
        classes_count=3, l2_leaf_reg=5, min_data_in_leaf=10,
        random_strength=1.5, bagging_temperature=0.8,
        early_stopping_rounds=40, random_seed=42,
        verbose=0, thread_count=-1,
    )
    model.fit(Pool(X_tr, y_tr, weight=sw),
              eval_set=Pool(X_val, y_val), use_best_model=True)
    return model


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

DARK_BG   = "#f8f5f0"
GRID_CLR  = "#e8e2d8"
TEXT_CLR  = "#4a4035"
UP_CLR    = "#1a7a3f"
DN_CLR    = "#c0392b"
VOL_CLR   = "#3a6ea5"

def make_price_volume_chart(df_1m: pd.DataFrame,
                             title: str = "CBRS — 1m Price & Volume") -> go.Figure:
    """
    Use integer bar index on X-axis to eliminate overnight / weekend gaps.
    Tick labels show actual ET timestamps at ~30-min intervals.
    """
    # ── Filter regular session ──
    reg = df_1m[
        (df_1m.index.time >= pd.Timestamp("09:30").time()) &
        (df_1m.index.time <= pd.Timestamp("16:00").time())
    ].copy()
    if reg.empty:
        reg = df_1m.copy()

    # ── Integer index (no time gaps) ──
    n   = len(reg)
    idx = list(range(n))

    # ── X-axis tick labels every 30 bars ──
    tick_step  = 30
    tickvals   = list(range(0, n, tick_step))
    ticklabels = [
        reg.index[i].strftime("%m/%d %H:%M")
        for i in tickvals
    ]

    # ── Day boundary vertical lines (where date changes) ──
    day_boundaries = []
    for i in range(1, n):
        if reg.index[i].date() != reg.index[i - 1].date():
            day_boundaries.append(i)

    # ── Candle colours ──
    colors = [UP_CLR if r >= 0 else DN_CLR
              for r in reg["close"].pct_change().fillna(0)]

    # ── Indicators ──
    tp    = (reg["high"] + reg["low"] + reg["close"]) / 3
    dk    = reg.index.normalize()
    vwap  = (tp * reg["volume"]).groupby(dk).cumsum() / (reg["volume"].groupby(dk).cumsum() + 1e-9)
    ema9  = reg["close"].ewm(span=9,  adjust=False).mean()
    ema21 = reg["close"].ewm(span=21, adjust=False).mean()

    delta = reg["close"].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

    vol_30m         = reg["volume"].resample("30min").sum()
    vol_30m_aligned = vol_30m.reindex(reg.index, method="ffill").values

    # ── Figure ──
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.02,
        subplot_titles=("", "", ""),
    )

    # Row 1 — Price
    fig.add_trace(go.Candlestick(
        x=idx,
        open=reg["open"].values, high=reg["high"].values,
        low=reg["low"].values,   close=reg["close"].values,
        increasing_line_color=UP_CLR, decreasing_line_color=DN_CLR,
        increasing_fillcolor=UP_CLR,  decreasing_fillcolor=DN_CLR,
        name="Price", line_width=1,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx, y=vwap.values, name="VWAP",
        line=dict(color="#b8860b", width=1.5, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=ema9.values, name="EMA9",
        line=dict(color="#3a6ea5", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=ema21.values, name="EMA21",
        line=dict(color="#7b4fa6", width=1),
    ), row=1, col=1)

    # Row 2 — RSI
    fig.add_trace(go.Scatter(
        x=idx, y=rsi.values, name="RSI",
        line=dict(color="#3a6ea5", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#c0392b", line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#1a7a3f", line_width=0.8, row=2, col=1)

    # Row 3 — Volume
    fig.add_trace(go.Bar(
        x=idx, y=reg["volume"].values, name="Volume",
        marker_color=colors, opacity=0.70,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=vol_30m_aligned,
        name="Vol 30m avg", fill="tozeroy",
        fillcolor="rgba(58,110,165,0.08)",
        line=dict(color=VOL_CLR, width=1, dash="dot"),
    ), row=3, col=1)

    # ── Day boundary lines across all rows ──
    for b in day_boundaries:
        for row in [1, 2, 3]:
            fig.add_vline(
                x=b, line_dash="dash",
                line_color="#a09080", line_width=1.2,
                row=row, col=1,
            )
        # Date label at top
        fig.add_annotation(
            x=b + 2, y=1, yref="paper",
            text=reg.index[b].strftime("%b %d"),
            showarrow=False,
            font=dict(size=10, color="#8a7968", family="Georgia, serif"),
            xanchor="left",
        )

    # ── Layout ──
    axis_common = dict(
        tickvals=tickvals,
        ticktext=ticklabels,
        tickangle=-45,
        tickfont=dict(size=9, family="Space Mono"),
        gridcolor=GRID_CLR,
        zeroline=False,
        showspikes=True,
        spikecolor="#a09080",
        spikethickness=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Georgia, serif", size=14, color="#1a1a2e")),
        paper_bgcolor=DARK_BG,
        plot_bgcolor="#ffffff",
        font=dict(family="Georgia, serif", color=TEXT_CLR),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd5c8", borderwidth=1,
                    font=dict(size=11)),
        margin=dict(l=60, r=20, t=50, b=60),
        height=700,
        xaxis =dict(**axis_common),
        xaxis2=dict(**axis_common),
        xaxis3=dict(**axis_common),
    )
    fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False)
    fig.update_yaxes(title_text="Price",  row=1, col=1, title_font_size=11)
    fig.update_yaxes(title_text="RSI",    row=2, col=1, title_font_size=11, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1, title_font_size=11)
    return fig


def make_volume_profile_chart(df_1m: pd.DataFrame) -> go.Figure:
    reg = df_1m[
        (df_1m.index.time >= pd.Timestamp("09:30").time()) &
        (df_1m.index.time <= pd.Timestamp("16:00").time())
    ].copy()
    intervals = (reg.index.hour * 60 + reg.index.minute - 570) // 30
    reg["interval_30m"] = np.clip(intervals, 0, 12)

    LABELS = {
        0:"09:30",1:"10:00",2:"10:30",3:"11:00",4:"11:30",5:"12:00",
        6:"12:30",7:"13:00",8:"13:30",9:"14:00",10:"14:30",11:"15:00",12:"15:30"
    }
    dates = sorted(reg.index.normalize().unique())

    fig = go.Figure()
    palette = ["#3b82f6", "#22c55e", "#f59e0b", "#c084fc"]

    for idx, d in enumerate(dates):
        day_df  = reg[reg.index.normalize() == d]
        grp     = day_df.groupby("interval_30m")["volume"].sum().reset_index()
        total   = grp["volume"].sum()
        grp["pct"] = grp["volume"] / (total + 1e-9) * 100
        grp["label"] = grp["interval_30m"].map(LABELS)

        fig.add_trace(go.Bar(
            x=grp["label"], y=grp["volume"],
            name=str(d.date()),
            marker_color=palette[idx % len(palette)],
            opacity=0.85,
            text=grp["pct"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(size=10, color=TEXT_CLR),
        ))

    fig.update_layout(
        title=dict(text="Volume by 30-min Interval",
                   font=dict(family="Georgia, serif", size=13, color="#1a1a2e")),
        paper_bgcolor=DARK_BG, plot_bgcolor="#ffffff",
        font=dict(family="Georgia, serif", color=TEXT_CLR),
        barmode="group",
        xaxis=dict(gridcolor=GRID_CLR, title="Time (ET)"),
        yaxis=dict(gridcolor=GRID_CLR, title="Volume"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ddd5c8", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=40),
        height=380,
    )
    return fig


def make_probability_gauge(proba: dict) -> go.Figure:
    labels = ["Crash", "Squeeze", "Breakout"]
    values = [proba["crash"], proba["squeeze"], proba["breakout"]]
    colors = [DN_CLR, "#b8860b", UP_CLR]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(size=13, family="Space Mono", color="#2c2c2c"),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor="#ffffff",
        font=dict(family="Georgia, serif", color=TEXT_CLR),
        xaxis=dict(range=[0,1], gridcolor=GRID_CLR,
                   tickformat=".0%", tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=12)),
        margin=dict(l=20, r=80, t=20, b=20),
        height=200,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# PROPHET ENGINE
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def run_prophet(
    _df_1m: pd.DataFrame,
    periods: int = 60,
    interval_width: float = 0.80,
    cache_key: str = "",   # ← 移除底線，Streamlit 才會納入 hash
) -> dict | None:
    """
    Fit Prophet on 1m close prices (regular session only).
    Returns forecast DataFrame + component dict.

    Prophet treats each bar as equally spaced — intraday gaps
    (overnight, weekend) are handled via make_future_dataframe
    with freq='T' (minutes), which is fine for within-day forecasts.
    """
    if not PROPHET_AVAILABLE:
        return None

    reg = _df_1m[
        (_df_1m.index.time >= pd.Timestamp("09:30").time()) &
        (_df_1m.index.time <= pd.Timestamp("16:00").time())
    ].copy()

    if len(reg) < 60:
        return None

    # Prophet requires tz-naive ds column
    prophet_df = pd.DataFrame({
        "ds": reg.index.tz_localize(None),
        "y" : reg["close"].values,
    }).dropna()

    # Add volume as regressor (normalised)
    vol_norm = reg["volume"].values / (reg["volume"].mean() + 1e-9)
    prophet_df["volume_norm"] = vol_norm

    m = Prophet(
        changepoint_prior_scale  = 0.3,    # flexibility of trend changes
        seasonality_prior_scale  = 0.1,
        interval_width           = interval_width,
        daily_seasonality        = False,
        weekly_seasonality       = False,
        yearly_seasonality       = False,
    )
    # Intraday "hourly" seasonality — period=390min (6.5h trading day)
    m.add_seasonality(
        name   = "intraday",
        period = 390 / (60 * 24),   # in days
        fourier_order = 8,
    )
    m.add_regressor("volume_norm", standardize=True)

    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    m.fit(prophet_df)

    # Future dataframe: extend by `periods` minutes
    last_ts   = prophet_df["ds"].iloc[-1]
    future_ts = pd.date_range(
        start  = last_ts + pd.Timedelta(minutes=1),
        periods= periods,
        freq   = "1min",
    )
    future = m.make_future_dataframe(periods=periods, freq="1min")
    # Fill volume_norm for future rows with recent mean
    future["volume_norm"] = vol_norm[-20:].mean()

    forecast = m.predict(future)

    # Separate historical fit vs future forecast
    hist_len = len(prophet_df)
    fc_hist  = forecast.iloc[:hist_len].copy()
    fc_fut   = forecast.iloc[hist_len:].copy()

    # Map back to ET timestamps for display
    fc_hist["ds_et"] = pd.to_datetime(fc_hist["ds"]).dt.tz_localize(ET)
    fc_fut["ds_et"]  = pd.to_datetime(fc_fut["ds"]).dt.tz_localize(ET)

    # Prophet signal: last actual vs predicted next bar
    last_actual   = prophet_df["y"].iloc[-1]
    next_yhat     = fc_fut["yhat"].iloc[0] if len(fc_fut) else last_actual
    next_yhat_lo  = fc_fut["yhat_lower"].iloc[0] if len(fc_fut) else last_actual
    next_yhat_hi  = fc_fut["yhat_upper"].iloc[0] if len(fc_fut) else last_actual

    trend_chg = (next_yhat - last_actual) / (last_actual + 1e-9)
    if   trend_chg >  0.003: prophet_signal = "📈 UP"
    elif trend_chg < -0.003: prophet_signal = "📉 DOWN"
    else:                    prophet_signal = "➡️  FLAT"

    return {
        "fc_hist"       : fc_hist,
        "fc_fut"        : fc_fut,
        "actual_df"     : prophet_df,
        "last_actual"   : last_actual,
        "next_yhat"     : next_yhat,
        "next_yhat_lo"  : next_yhat_lo,
        "next_yhat_hi"  : next_yhat_hi,
        "trend_chg_pct" : trend_chg * 100,
        "prophet_signal": prophet_signal,
        "periods"       : periods,
    }


def make_prophet_chart(result: dict, ticker: str) -> go.Figure:
    """
    Prophet forecast chart using integer bar index on X-axis,
    eliminating overnight / weekend gaps from the display.
    Historical bars + forecast bars are concatenated into one
    continuous sequence; a vertical line marks the boundary.
    """
    fc_hist = result["fc_hist"].reset_index(drop=True)
    fc_fut  = result["fc_fut"].reset_index(drop=True)
    actual  = result["actual_df"].reset_index(drop=True)
    periods = result["periods"]

    n_hist = len(actual)        # number of historical bars
    n_fut  = len(fc_fut)        # number of forecast bars
    n_total = n_hist + n_fut

    # ── Integer indices ──
    idx_hist = list(range(n_hist))
    idx_fut  = list(range(n_hist, n_total))

    # ── Build unified tick labels every 30 bars ──
    tick_step = 30
    all_ds = list(actual["ds"]) + list(fc_fut["ds"]) if n_fut else list(actual["ds"])
    tickvals  = list(range(0, n_total, tick_step))
    ticklabels = []
    for i in tickvals:
        if i < len(all_ds):
            ts = pd.to_datetime(all_ds[i])
            ticklabels.append(ts.strftime("%m/%d %H:%M"))
        else:
            ticklabels.append("")

    # ── Day boundary lines within historical section ──
    day_boundaries = []
    for i in range(1, n_hist):
        d_prev = pd.to_datetime(actual["ds"].iloc[i - 1]).date()
        d_curr = pd.to_datetime(actual["ds"].iloc[i]).date()
        if d_curr != d_prev:
            day_boundaries.append(i)

    fig = go.Figure()

    # ── Actual close ──
    fig.add_trace(go.Scatter(
        x=idx_hist, y=actual["y"].values,
        name="Actual Close",
        line=dict(color="#2c2c2c", width=1.5),
        opacity=0.9,
    ))

    # ── Prophet fitted line ──
    fig.add_trace(go.Scatter(
        x=idx_hist, y=fc_hist["yhat"].values,
        name="Prophet Fit",
        line=dict(color="#7b4fa6", width=1.5, dash="dot"),
    ))

    # ── Historical confidence band ──
    x_band_h = idx_hist + idx_hist[::-1]
    y_band_h = (list(fc_hist["yhat_upper"].values) +
                list(fc_hist["yhat_lower"].values[::-1]))
    fig.add_trace(go.Scatter(
        x=x_band_h, y=y_band_h,
        fill="toself",
        fillcolor="rgba(123,79,166,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Fit Band", showlegend=False,
    ))

    if n_fut > 0:
        # ── Connector: last hist bar → first forecast bar ──
        fig.add_trace(go.Scatter(
            x=[n_hist - 1, n_hist],
            y=[fc_hist["yhat"].iloc[-1], fc_fut["yhat"].iloc[0]],
            line=dict(color="#3a6ea5", width=2),
            showlegend=False,
        ))

        # ── Forecast line ──
        fig.add_trace(go.Scatter(
            x=idx_fut, y=fc_fut["yhat"].values,
            name=f"Forecast (+{periods}m)",
            line=dict(color="#3a6ea5", width=2.5),
            mode="lines",
        ))

        # ── Forecast CI cone ──
        x_cone = idx_fut + idx_fut[::-1]
        y_cone = (list(fc_fut["yhat_upper"].values) +
                  list(fc_fut["yhat_lower"].values[::-1]))
        fig.add_trace(go.Scatter(
            x=x_cone, y=y_cone,
            fill="toself",
            fillcolor="rgba(58,110,165,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(result.get('interval_width', 0.8) * 100)}% CI",
        ))

        # ── Next-bar target marker ──
        fig.add_trace(go.Scatter(
            x=[n_hist],
            y=[result["next_yhat"]],
            mode="markers+text",
            marker=dict(size=10, color="#3a6ea5",
                        line=dict(color="#f8f5f0", width=2)),
            text=[f"  ${result['next_yhat']:.2f}"],
            textfont=dict(color="#3a6ea5", size=12, family="Space Mono"),
            textposition="middle right",
            name="Next bar target",
        ))

        # ── Hist / Forecast boundary line ──
        fig.add_vline(
            x=n_hist - 0.5,
            line_dash="dash", line_color="#a09080", line_width=1.2,
            annotation_text=" Forecast →",
            annotation_font_color="#8a7968",
            annotation_font_size=11,
            annotation_position="top right",
        )

    # ── Day boundary lines ──
    for b in day_boundaries:
        fig.add_vline(
            x=b - 0.5,
            line_dash="dot", line_color="#c8bfb4", line_width=1,
        )
        fig.add_annotation(
            x=b + 1, y=1, yref="paper",
            text=pd.to_datetime(actual["ds"].iloc[b]).strftime("%b %d"),
            showarrow=False,
            font=dict(size=10, color="#8a7968", family="Georgia, serif"),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Prophet Forecast  (intraday · no gap)",
            font=dict(family="Georgia, serif", size=13, color="#1a1a2e"),
        ),
        paper_bgcolor=DARK_BG, plot_bgcolor="#ffffff",
        font=dict(family="Georgia, serif", color=TEXT_CLR),
        xaxis=dict(
            tickvals=tickvals, ticktext=ticklabels,
            tickangle=-45,
            tickfont=dict(size=9, family="Space Mono"),
            gridcolor=GRID_CLR, zeroline=False,
            showspikes=True, spikecolor="#a09080", spikethickness=1,
        ),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False, title="Price (USD)"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd5c8", borderwidth=1,
                    font=dict(size=11)),
        margin=dict(l=60, r=20, t=55, b=60),
        height=500,
        hovermode="x unified",
    )
    return fig


def make_prophet_components_chart(result: dict) -> go.Figure:
    """
    Prophet trend + seasonality components — integer index,
    no non-trading gaps.
    """
    fc_hist = result["fc_hist"].reset_index(drop=True)
    n       = len(fc_hist)
    idx     = list(range(n))

    # Tick labels every 30 bars
    tick_step  = 30
    tickvals   = list(range(0, n, tick_step))
    ticklabels = [
        pd.to_datetime(fc_hist["ds"].iloc[i]).strftime("%m/%d %H:%M")
        for i in tickvals
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Trend Component", "Intraday Seasonality"],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=idx, y=fc_hist["trend"].values,
        name="Trend", line=dict(color="#b8860b", width=1.5),
    ), row=1, col=1)

    if "additive_terms" in fc_hist.columns:
        fig.add_trace(go.Scatter(
            x=idx, y=fc_hist["additive_terms"].values,
            name="Seasonality", line=dict(color="#7b4fa6", width=1.5),
            fill="tozeroy", fillcolor="rgba(123,79,166,0.10)",
        ), row=2, col=1)

    axis_common = dict(
        tickvals=tickvals, ticktext=ticklabels,
        tickangle=-45,
        tickfont=dict(size=9, family="Space Mono"),
        gridcolor=GRID_CLR,
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor="#ffffff",
        font=dict(family="Georgia, serif", color=TEXT_CLR),
        xaxis =dict(**axis_common),
        xaxis2=dict(**axis_common),
        yaxis =dict(gridcolor=GRID_CLR, title="Price"),
        yaxis2=dict(gridcolor=GRID_CLR, title="Effect"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd5c8", borderwidth=1),
        margin=dict(l=60, r=20, t=40, b=60),
        height=400,
    )
    fig.update_yaxes(gridcolor=GRID_CLR)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    ticker = st.text_input("Ticker", value="CBRS").upper().strip()

    today      = date.today()
    # ── 修正：移除寫死的 ipo_date，改為動態 min（7天前，yfinance 1m 上限）──
    min_date   = today - timedelta(days=7)
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=date(2010, 1, 1),   # 任意過去日期皆可輸入
        max_value=today,
        help="yfinance 1m data: max 7 calendar days back",
    )
    end_date = st.date_input(
        "End Date",
        value=today,
        min_value=start_date,
        max_value=today + timedelta(days=1),
    )

    st.markdown("---")
    st.markdown("**Model Parameters**")
    horizon   = st.slider("Label Horizon (bars)", 5, 30, 10)
    bt_thresh = st.slider("Breakout Threshold %", 0.2, 2.0, 0.6, step=0.1) / 100
    ct_thresh = st.slider("Crash Threshold %",   -2.0, -0.2, -0.6, step=0.1) / 100
    conf_thr  = st.slider("Confidence Gate %",   30, 70, 45) / 100

    st.markdown("---")
    st.markdown("**Prophet Settings**")
    use_prophet     = st.toggle("Enable Prophet Forecast", value=True,
                                disabled=not PROPHET_AVAILABLE,
                                help="Requires `pip install prophet`")
    prophet_periods = st.slider("Forecast Horizon (bars)", 10, 120, 60,
                                disabled=not use_prophet)
    prophet_ci      = st.slider("Confidence Interval %", 50, 95, 80,
                                disabled=not use_prophet) / 100

    if not PROPHET_AVAILABLE:
        st.caption("⚠️ `prophet` not installed — run `pip install prophet`")

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

st.markdown("# 🧠 CBRS Multi-Timeframe Predictor")
st.markdown(
    "<span style='font-family:Space Mono;font-size:13px;color:#64748b'>"
    "CatBoost · Prophet · 1m / 5m / 15m / 30m · Breakout / Squeeze / Crash</span>",
    unsafe_allow_html=True,
)

# ── Feature Summary Panel (always visible) ──
st.markdown("""
<style>
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 18px 0 8px 0;
}
.summary-block {
    background: linear-gradient(135deg, #ffffff 0%, #f9f5ef 100%);
    border: 1px solid #ddd5c8;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.summary-block-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3a6ea5;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.summary-block-title .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3a6ea5;
    display: inline-block;
}
.summary-item {
    font-family: Georgia, serif;
    font-size: 12.5px;
    color: #6b5e52;
    padding: 4px 0;
    border-bottom: 1px solid #ede8e0;
    display: flex;
    justify-content: space-between;
}
.summary-item:last-child { border-bottom: none; }
.summary-item b { color: #2c2c2c; font-weight: 600; }
.summary-note {
    font-family: Georgia, serif;
    font-size: 11.5px;
    color: #7a6e62;
    margin-top: 14px;
    padding: 10px 14px;
    background: #fff8f0;
    border-left: 3px solid #c8a882;
    border-radius: 0 6px 6px 0;
}
</style>

<div class="summary-grid">

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot"></span> 📥 Input Data</div>
    <div class="summary-item"><span>Timeframe</span><b>1-minute OHLCV</b></div>
    <div class="summary-item"><span>Pre-market</span><b>Volume included</b></div>
    <div class="summary-item"><span>Session</span><b>09:30–16:00 ET</b></div>
    <div class="summary-item"><span>Source</span><b>yfinance (≤7 days)</b></div>
    <div class="summary-item"><span>Date select</span><b>Sidebar — any range</b></div>
  </div>

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot" style="background:#818cf8"></span> ⚙️ Multi-Scale Features</div>
    <div class="summary-item"><span>1m</span><b>VWAP · Spread · Imbalance · Candle</b></div>
    <div class="summary-item"><span>5m</span><b>ATR · RSI · Vol Spike · EMA Cross</b></div>
    <div class="summary-item"><span>15m</span><b>BB Squeeze · VWAP Dist · Momentum</b></div>
    <div class="summary-item"><span>30m</span><b>EMA Accel · Vol Trend · Regime</b></div>
    <div class="summary-item"><span>Time</span><b>sin/cos · First30m · Power Hour</b></div>
  </div>

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot" style="background:#22c55e"></span> 🎯 CatBoost Model</div>
    <div class="summary-item"><span>Algorithm</span><b>CatBoostClassifier</b></div>
    <div class="summary-item"><span>Loss</span><b>MultiClass (Logloss)</b></div>
    <div class="summary-item"><span>Output</span><b>Breakout / Squeeze / Crash %</b></div>
    <div class="summary-item"><span>Label</span><b>Forward ±0.6% over N bars</b></div>
    <div class="summary-item"><span>Guard</span><b>Confidence gate (default 45%)</b></div>
  </div>

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot" style="background:#f59e0b"></span> 📊 Charts</div>
    <div class="summary-item"><span>Price chart</span><b>Candlestick + VWAP + EMA9/21</b></div>
    <div class="summary-item"><span>Oscillator</span><b>RSI(14) with OB/OS lines</b></div>
    <div class="summary-item"><span>Volume</span><b>1m bar + 30m rolling profile</b></div>
    <div class="summary-item"><span>Vol profile</span><b>30-min interval % breakdown</b></div>
    <div class="summary-item"><span>Filter</span><b>Select date or view All</b></div>
  </div>

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot" style="background:#c084fc"></span> 🔮 Prophet Forecast</div>
    <div class="summary-item"><span>Model</span><b>Facebook Prophet</b></div>
    <div class="summary-item"><span>Seasonality</span><b>Intraday (390-min cycle)</b></div>
    <div class="summary-item"><span>Regressor</span><b>Normalised volume</b></div>
    <div class="summary-item"><span>Output</span><b>Price cone + CI band</b></div>
    <div class="summary-item"><span>Signal</span><b>UP / DOWN / FLAT direction</b></div>
  </div>

  <div class="summary-block">
    <div class="summary-block-title"><span class="dot" style="background:#ef4444"></span> 🤝 Model Agreement</div>
    <div class="summary-item"><span>✅ Agree</span><b>Signal strengthened</b></div>
    <div class="summary-item"><span>⚠️ Disagree</span><b>Exercise caution</b></div>
    <div class="summary-item"><span>⚪ Uncertain</span><b>Defer to Prophet</b></div>
    <div class="summary-item"><span>Feature imp.</span><b>Top-20 bar chart</b></div>
    <div class="summary-item"><span>Components</span><b>Trend + seasonality</b></div>
  </div>

</div>

<div class="summary-note">
  ⚠️ &nbsp;For <b>research reference only</b> — not financial advice.
  With only 2 trading days of data, model confidence is inherently limited.
  Use signals as one input among many, not as standalone trade triggers.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if not run_btn:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis**.")
    st.stop()

# ── Fetch ──
start_str = start_date.strftime("%Y-%m-%d")
end_str   = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker} 1m data …"):
    df_1m = fetch_1m(ticker, start_str, end_str)

if df_1m.empty:
    st.error(f"❌ No data found for **{ticker}**. Try a different ticker or date range.")
    st.stop()

dates_found = sorted(set(df_1m.index.date))
st.success(f"✅ Loaded **{len(df_1m):,}** bars  |  Dates: {[str(d) for d in dates_found]}")

# ── Date filter for display ──
col_a, col_b = st.columns([3, 1])
with col_a:
    if len(dates_found) > 1:
        selected_date = st.selectbox(
            "📅 Chart Date",
            options=["All"] + [str(d) for d in dates_found],
            index=0,
        )
    else:
        selected_date = str(dates_found[0])

if selected_date == "All":
    df_plot = df_1m
else:
    df_plot = df_1m[df_1m.index.date == date.fromisoformat(selected_date)]

# ── Price + Volume Chart ──
st.plotly_chart(
    make_price_volume_chart(df_plot, f"{ticker} — {selected_date}  (1m)"),
    use_container_width=True,
)

# ── Volume Profile ──
st.plotly_chart(
    make_volume_profile_chart(df_1m),
    use_container_width=True,
)

st.markdown("---")

# ── Model ──
with st.spinner("Building multi-timeframe features …"):
    master = build_feature_matrix(df_1m)

with st.spinner("Training CatBoostClassifier …"):
    master["label"] = label_bars(master, horizon=horizon,
                                  bt=bt_thresh, ct=ct_thresh)
    labeled   = master.dropna(subset=["label"])
    feat_cols = get_feat_cols(labeled)
    X = labeled[feat_cols]
    y = labeled["label"].astype(int)

    lbl_counts = y.value_counts()
    c0 = int(lbl_counts.get(0, 0))
    c1 = int(lbl_counts.get(1, 0))
    c2 = int(lbl_counts.get(2, 0))

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Crash bars",   c0)
    col2.metric("🟡 Squeeze bars", c1)
    col3.metric("🟢 Breakout bars",c2)

    if len(X) < 50 or len(y.unique()) < 2:
        st.warning("⚠️ Insufficient data to train model. "
                   "Try extending the date range.")
        st.stop()

    model = train_model(
        # ── 修正：cache key 含 ticker + date range，換 ticker/日期強制重訓 ──
        hash(f"{ticker}|{start_str}|{end_str}|" + str(X.values.tobytes()[:512])),
        X, y
    )

if model is None:
    st.error("Model training failed.")
    st.stop()

# ── Prediction ──
st.markdown("## 🎯 Latest Bar Prediction")

last_feat = master[feat_cols].iloc[[-1]]
proba     = model.predict_proba(last_feat)[0]
max_p     = proba.max()
trend_idx = int(np.argmax(proba))
trend_map = {0:"CRASH", 1:"SQUEEZE", 2:"BREAKOUT"}
color_map = {0:"trend-crash", 1:"trend-squeeze", 2:"trend-breakout"}
trend     = trend_map[trend_idx] if max_p >= conf_thr else "UNCERTAIN"
tclass    = color_map[trend_idx] if max_p >= conf_thr else "trend-uncertain"
valid     = max_p >= conf_thr

proba_dict = {"crash": proba[0], "squeeze": proba[1], "breakout": proba[2]}

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f"""<div class="metric-card">
  <div class="metric-label">Last Close</div>
  <div class="metric-value" style="color:#1a1a2e">${master['close'].iloc[-1]:.2f}</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""<div class="metric-card">
  <div class="metric-label">Breakout</div>
  <div class="metric-value trend-breakout">{proba[2]:.1%}</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""<div class="metric-card">
  <div class="metric-label">Squeeze</div>
  <div class="metric-value trend-squeeze">{proba[1]:.1%}</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""<div class="metric-card">
  <div class="metric-label">Crash</div>
  <div class="metric-value trend-crash">{proba[0]:.1%}</div>
</div>""", unsafe_allow_html=True)

c5.markdown(f"""<div class="metric-card">
  <div class="metric-label">Trend</div>
  <div class="metric-value {tclass}">{trend}</div>
  <div style="margin-top:6px">
    <span class="signal-badge {'badge-valid' if valid else 'badge-invalid'}">
      {'✓ VALID' if valid else '✗ UNCERTAIN'}
    </span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(make_probability_gauge(proba_dict), use_container_width=True)

# ─────────────────────────────────────────────
# PROPHET SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔮 Prophet Forecast  <span style='font-size:13px;color:#8a7968;font-family:Georgia,serif'>for reference only</span>", unsafe_allow_html=True)

if not PROPHET_AVAILABLE:
    st.warning("Prophet is not installed. Run `pip install prophet` to enable this section.")
elif not use_prophet:
    st.info("Prophet forecast is disabled. Toggle it on in the sidebar.")
else:
    with st.spinner("Fitting Prophet model …"):
        prophet_result = run_prophet(
            df_1m,
            periods        = prophet_periods,
            interval_width = prophet_ci,
            cache_key      = f"{ticker}|{start_str}|{end_str}",
        )

    if prophet_result is None:
        st.warning("⚠️ Not enough data to fit Prophet (need ≥ 60 regular-session bars).")
    else:
        # ── Signal summary cards ──
        pa, pb, pc, pd_ = st.columns(4)
        last_p  = prophet_result["last_actual"]
        next_p  = prophet_result["next_yhat"]
        chg_pct = prophet_result["trend_chg_pct"]
        sig     = prophet_result["prophet_signal"]
        sig_color = "#1a7a3f" if "UP" in sig else "#c0392b" if "DOWN" in sig else "#b8860b"

        pa.markdown(f"""<div class="metric-card">
          <div class="metric-label">Last Close</div>
          <div class="metric-value" style="color:#1a1a2e">${last_p:.2f}</div>
        </div>""", unsafe_allow_html=True)

        pb.markdown(f"""<div class="metric-card">
          <div class="metric-label">Next Bar Target</div>
          <div class="metric-value" style="color:#3a6ea5">${next_p:.2f}</div>
        </div>""", unsafe_allow_html=True)

        pc.markdown(f"""<div class="metric-card">
          <div class="metric-label">Expected Δ</div>
          <div class="metric-value" style="color:{sig_color}">{chg_pct:+.3f}%</div>
        </div>""", unsafe_allow_html=True)

        pd_.markdown(f"""<div class="metric-card">
          <div class="metric-label">Prophet Signal</div>
          <div class="metric-value" style="color:{sig_color};font-size:22px">{sig}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CI range info ──
        lo = prophet_result["next_yhat_lo"]
        hi = prophet_result["next_yhat_hi"]
        st.markdown(
            f"<div style='font-family:Space Mono;font-size:12px;color:#8a7968;"
            f"text-align:center;padding:6px 0'>"
            f"{int(prophet_ci*100)}% Confidence Interval for next bar:  "
            f"<span style='color:#1a1a2e'>${lo:.2f}</span> — "
            f"<span style='color:#1a1a2e'>${hi:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Main Prophet chart ──
        st.plotly_chart(
            make_prophet_chart(prophet_result, ticker),
            use_container_width=True,
        )

        # ── Components chart ──
        with st.expander("📈 Prophet Components (Trend + Seasonality)", expanded=False):
            st.plotly_chart(
                make_prophet_components_chart(prophet_result),
                use_container_width=True,
            )

        # ── Agreement with CatBoost ──
        st.markdown("#### 🤝 Model Agreement")
        cb_trend  = trend      # from CatBoost section above
        ph_signal = sig

        agree = (
            ("BREAKOUT" in cb_trend and "UP"   in ph_signal) or
            ("CRASH"    in cb_trend and "DOWN" in ph_signal) or
            ("SQUEEZE"  in cb_trend and "FLAT" in ph_signal)
        )
        if "UNCERTAIN" in cb_trend:
            agree_txt   = "⚪ CatBoost UNCERTAIN — defer to Prophet"
            agree_color = "#94a3b8"
        elif agree:
            agree_txt   = "✅ Both models AGREE — signal strengthened"
            agree_color = "#22c55e"
        else:
            agree_txt   = "⚠️ Models DISAGREE — exercise caution"
            agree_color = "#f59e0b"

        st.markdown(
            f"<div style='background:linear-gradient(135deg,#ffffff,#f5f0e8);"
            f"border:1.5px solid {agree_color};border-radius:10px;padding:14px 20px;"
            f"font-family:Georgia,serif;font-size:13px;color:{agree_color};text-align:center;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.06)'>"
            f"CatBoost: <b>{cb_trend}</b> &nbsp;|&nbsp; Prophet: <b>{ph_signal}</b>"
            f"<br><span style='font-size:15px;margin-top:6px;display:block'>{agree_txt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Importance ──
st.markdown("---")
st.markdown("## 📊 Top-20 Feature Importance")

fi = pd.Series(model.get_feature_importance(), index=feat_cols).sort_values(ascending=False)
fi_top = fi.head(20)

fig_fi = go.Figure(go.Bar(
    x=fi_top.values[::-1],
    y=fi_top.index[::-1],
    orientation="h",
    marker=dict(
        color=fi_top.values[::-1],
        colorscale=[[0,"#d4e8f5"],[0.5,"#3a6ea5"],[1,"#1a3a6e"]],
        showscale=False,
    ),
    text=[f"{v:.2f}" for v in fi_top.values[::-1]],
    textposition="outside",
    textfont=dict(size=10, color=TEXT_CLR),
))
fig_fi.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor="#ffffff",
    font=dict(family="Georgia, serif", color=TEXT_CLR),
    xaxis=dict(gridcolor=GRID_CLR, title="Importance Score"),
    yaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=11)),
    margin=dict(l=20, r=80, t=20, b=20),
    height=500,
)
st.plotly_chart(fig_fi, use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#c8bfb4;font-family:Georgia,serif;"
    "font-size:11px;padding:12px'>CBRS MTF Predictor · For research only · "
    "Not financial advice</div>",
    unsafe_allow_html=True,
)
