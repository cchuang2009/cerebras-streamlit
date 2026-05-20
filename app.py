"""
app.py — Multi-Timeframe Trading Predictor (Streamlit)
────────────────────────────────────────────────────────
All heavy logic lives in mtf_modules/.
This file only handles UI layout and Streamlit state.

Run:  streamlit run app.py
Deps: pip install streamlit plotly yfinance catboost
      scikit-learn pytz pandas numpy prophet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import streamlit as st

# ── Module imports ──────────────────────────────────────
from mtf_modules.data     import (fetch_1m, fetch_spy_1m,
                                   fetch_daily, get_info,
                                   regular_session)
from mtf_modules.features import (build_master, get_feature_cols,
                                   marketcap_tier)
from mtf_modules.labels   import (make_labels, dynamic_confidence_gate,
                                   label_summary)
from mtf_modules.model    import (train, predict as model_predict,
                                   feature_importance)
from mtf_modules.prophet_model import (fit_prophet, model_agreement)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MTF Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: Georgia, 'Times New Roman', serif;
    background-color: #f8f5f0; color: #2c2c2c;
}
h1,h2,h3,h4 { font-family: Georgia, serif; color: #1a1a2e; font-weight:700; }
.main .block-container { background-color: #f8f5f0; padding-top:2rem; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#fffdf9 0%,#f3ede4 100%);
    border-right: 1px solid #ddd5c8;
}
.metric-card {
    background: linear-gradient(135deg,#ffffff 0%,#f5f0e8 100%);
    border:1px solid #ddd5c8; border-radius:12px; padding:16px 20px;
    text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
.metric-label {
    font-size:11px; text-transform:uppercase; letter-spacing:1.5px;
    color:#8a7968; font-family:'Space Mono',monospace;
}
.metric-value {
    font-size:26px; font-weight:700; font-family:'Space Mono',monospace;
    margin-top:4px; color:#1a1a2e;
}
.trend-breakout{color:#1a7a3f;} .trend-crash{color:#c0392b;}
.trend-squeeze{color:#b8860b;}  .trend-uncertain{color:#7a7060;}
.signal-badge {
    display:inline-block; padding:4px 14px; border-radius:999px;
    font-size:12px; font-family:'Space Mono',monospace; font-weight:700;
}
.badge-valid   {background:#dcf5e7;color:#1a7a3f;border:1px solid #1a7a3f;}
.badge-invalid {background:#f0ede8;color:#8a7968;border:1px solid #c8bfb4;}
hr { border-color:#ddd5c8; }
</style>
""", unsafe_allow_html=True)

DARK_BG  = "#f8f5f0"
GRID_CLR = "#e8e2d8"
TEXT_CLR = "#4a4035"
UP_CLR   = "#1a7a3f"
DN_CLR   = "#c0392b"
VOL_CLR  = "#3a6ea5"

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

def _tick_labels(index, step=30):
    n = len(index)
    vals = list(range(0, n, step))
    texts = [index[i].strftime("%m/%d %H:%M") for i in vals]
    return vals, texts

def _day_boundaries(index):
    return [i for i in range(1, len(index))
            if index[i].date() != index[i-1].date()]

def chart_price(df: pd.DataFrame, title: str) -> go.Figure:
    reg = regular_session(df)
    if reg.empty: reg = df
    n   = len(reg); idx = list(range(n))
    tv, tl = _tick_labels(reg.index)
    bounds = _day_boundaries(reg.index)
    colors = [UP_CLR if r >= 0 else DN_CLR
              for r in reg["close"].pct_change().fillna(0)]

    tp   = (reg["high"]+reg["low"]+reg["close"])/3
    dk   = reg.index.normalize()
    vwap = (tp*reg["volume"]).groupby(dk).cumsum()/(reg["volume"].groupby(dk).cumsum()+1e-9)
    ema9  = reg["close"].ewm(span=9,  adjust=False).mean()
    ema21 = reg["close"].ewm(span=21, adjust=False).mean()
    delta = reg["close"].diff()
    gain  = delta.clip(lower=0).rolling(14,min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14,min_periods=1).mean()
    rsi   = 100 - 100/(1+gain/(loss+1e-9))
    vol30 = reg["volume"].resample("30min").sum().reindex(reg.index,method="ffill").values

    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                        row_heights=[0.55,0.25,0.20],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=idx,open=reg["open"].values,
        high=reg["high"].values,low=reg["low"].values,close=reg["close"].values,
        increasing_line_color=UP_CLR,decreasing_line_color=DN_CLR,
        increasing_fillcolor=UP_CLR,decreasing_fillcolor=DN_CLR,
        name="Price",line_width=1),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=vwap.values,name="VWAP",
        line=dict(color="#b8860b",width=1.5,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema9.values,name="EMA9",
        line=dict(color="#3a6ea5",width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema21.values,name="EMA21",
        line=dict(color="#7b4fa6",width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=rsi.values,name="RSI",
        line=dict(color="#3a6ea5",width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_dash="dot",line_color=DN_CLR,line_width=0.8,row=2,col=1)
    fig.add_hline(y=30,line_dash="dot",line_color=UP_CLR,line_width=0.8,row=2,col=1)
    fig.add_trace(go.Bar(x=idx,y=reg["volume"].values,name="Volume",
        marker_color=colors,opacity=0.70),row=3,col=1)
    fig.add_trace(go.Scatter(x=idx,y=vol30,name="Vol 30m",fill="tozeroy",
        fillcolor="rgba(58,110,165,0.08)",
        line=dict(color=VOL_CLR,width=1,dash="dot")),row=3,col=1)

    for b in bounds:
        for r in [1,2,3]:
            fig.add_vline(x=b,line_dash="dash",line_color="#a09080",
                          line_width=1.2,row=r,col=1)
        fig.add_annotation(x=b+2,y=1,yref="paper",
            text=reg.index[b].strftime("%b %d"),showarrow=False,
            font=dict(size=10,color="#8a7968",family="Georgia,serif"),xanchor="left")

    ax = dict(tickvals=tv,ticktext=tl,tickangle=-45,
              tickfont=dict(size=9,family="Space Mono"),
              gridcolor=GRID_CLR,zeroline=False,
              showspikes=True,spikecolor="#a09080",spikethickness=1)
    fig.update_layout(
        title=dict(text=title,font=dict(family="Georgia,serif",size=14,color="#1a1a2e")),
        paper_bgcolor=DARK_BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT_CLR),
        xaxis_rangeslider_visible=False,height=700,
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1,font=dict(size=11)),
        margin=dict(l=60,r=20,t=50,b=60),
        xaxis=dict(**ax),xaxis2=dict(**ax),xaxis3=dict(**ax),
    )
    fig.update_yaxes(gridcolor=GRID_CLR,zeroline=False)
    fig.update_yaxes(title_text="Price",row=1,col=1,title_font_size=11)
    fig.update_yaxes(title_text="RSI",  row=2,col=1,title_font_size=11,range=[0,100])
    fig.update_yaxes(title_text="Volume",row=3,col=1,title_font_size=11)
    return fig


def chart_volume_profile(df: pd.DataFrame) -> go.Figure:
    reg = regular_session(df).copy()
    intervals = (reg.index.hour*60+reg.index.minute-570)//30
    reg["interval_30m"] = np.clip(intervals, 0, 12)
    LABELS = {0:"09:30",1:"10:00",2:"10:30",3:"11:00",4:"11:30",5:"12:00",
              6:"12:30",7:"13:00",8:"13:30",9:"14:00",10:"14:30",11:"15:00",12:"15:30"}
    palette = ["#3a6ea5","#1a7a3f","#b8860b","#7b4fa6"]
    fig = go.Figure()
    for idx_d, d in enumerate(sorted(reg.index.normalize().unique())):
        grp = reg[reg.index.normalize()==d].groupby("interval_30m")["volume"].sum().reset_index()
        total = grp["volume"].sum()
        grp["pct"] = grp["volume"]/(total+1e-9)*100
        grp["label"] = grp["interval_30m"].map(LABELS)
        vol_safe = grp["volume"].apply(lambda v: int(v) if pd.notna(v) else 0)
        pct_safe = grp["pct"].apply(lambda p: float(p) if pd.notna(p) else 0.0)
        fig.add_trace(go.Bar(
            x=grp["label"],y=vol_safe,name=str(d.date()),
            marker_color=palette[idx_d%len(palette)],opacity=0.85,
            text=pct_safe.map("{:.1f}%".format),textposition="outside",
            textfont=dict(size=10,color=TEXT_CLR),
        ))
    fig.update_layout(
        title=dict(text="Volume by 30-min Interval",font=dict(family="Georgia,serif",size=13,color="#1a1a2e")),
        paper_bgcolor=DARK_BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT_CLR),
        barmode="group",height=380,
        xaxis=dict(gridcolor=GRID_CLR,title="Time (ET)"),
        yaxis=dict(gridcolor=GRID_CLR,title="Volume"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1),
        margin=dict(l=60,r=20,t=50,b=40),
    )
    return fig


def chart_prophet(result: dict, ticker: str) -> go.Figure:
    fc_hist = result["fc_hist"].reset_index(drop=True)
    fc_fut  = result["fc_fut"].reset_index(drop=True)
    actual  = result["actual_df"].reset_index(drop=True)
    n_hist  = len(actual); n_fut = len(fc_fut)
    n_total = n_hist + n_fut
    idx_h   = list(range(n_hist)); idx_f = list(range(n_hist,n_total))
    all_ds  = list(actual["ds"]) + list(fc_fut["ds"]) if n_fut else list(actual["ds"])
    step    = 30
    tv = list(range(0,n_total,step))
    tl = [pd.to_datetime(all_ds[i]).strftime("%m/%d %H:%M")
          if i<len(all_ds) else "" for i in tv]
    bounds = []
    for i in range(1,n_hist):
        if pd.to_datetime(actual["ds"].iloc[i]).date() != pd.to_datetime(actual["ds"].iloc[i-1]).date():
            bounds.append(i)

    import pytz
    ET = pytz.timezone("America/New_York")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx_h,y=actual["y"].values,name="Actual",
        line=dict(color="#2c2c2c",width=1.5)))
    fig.add_trace(go.Scatter(x=idx_h,y=fc_hist["yhat"].values,name="Prophet Fit",
        line=dict(color="#7b4fa6",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(
        x=idx_h+idx_h[::-1],
        y=list(fc_hist["yhat_upper"].values)+list(fc_hist["yhat_lower"].values[::-1]),
        fill="toself",fillcolor="rgba(123,79,166,0.08)",
        line=dict(color="rgba(0,0,0,0)"),name="Fit Band",showlegend=False))
    if n_fut>0:
        fig.add_trace(go.Scatter(x=[n_hist-1,n_hist],
            y=[fc_hist["yhat"].iloc[-1],fc_fut["yhat"].iloc[0]],
            line=dict(color="#3a6ea5",width=2),showlegend=False))
        fig.add_trace(go.Scatter(x=idx_f,y=fc_fut["yhat"].values,
            name=f"Forecast (+{result['periods']}m)",
            line=dict(color="#3a6ea5",width=2.5)))
        fig.add_trace(go.Scatter(
            x=idx_f+idx_f[::-1],
            y=list(fc_fut["yhat_upper"].values)+list(fc_fut["yhat_lower"].values[::-1]),
            fill="toself",fillcolor="rgba(58,110,165,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(result['interval_width']*100)}% CI"))
        fig.add_trace(go.Scatter(x=[n_hist],y=[result["next_yhat"]],
            mode="markers+text",marker=dict(size=10,color="#3a6ea5",line=dict(color="#f8f5f0",width=2)),
            text=[f"  ${result['next_yhat']:.2f}"],
            textfont=dict(color="#3a6ea5",size=12,family="Space Mono"),
            textposition="middle right",name="Next bar"))
        fig.add_vline(x=n_hist-0.5,line_dash="dash",line_color="#a09080",line_width=1.2,
            annotation_text=" Forecast →",annotation_font_color="#8a7968",
            annotation_font_size=11,annotation_position="top right")
    for b in bounds:
        fig.add_vline(x=b-0.5,line_dash="dot",line_color="#c8bfb4",line_width=1)
    fig.update_layout(
        title=dict(text=f"{ticker} — Prophet Forecast",
                   font=dict(family="Georgia,serif",size=13,color="#1a1a2e")),
        paper_bgcolor=DARK_BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT_CLR),
        xaxis=dict(tickvals=tv,ticktext=tl,tickangle=-45,
                   tickfont=dict(size=9,family="Space Mono"),
                   gridcolor=GRID_CLR,zeroline=False,
                   showspikes=True,spikecolor="#a09080"),
        yaxis=dict(gridcolor=GRID_CLR,zeroline=False,title="Price"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1,font=dict(size=11)),
        margin=dict(l=60,r=20,t=55,b=60),height=500,hovermode="x unified",
    )
    return fig


def chart_proba(proba: dict) -> go.Figure:
    labels = ["Crash","Squeeze","Breakout"]
    values = [proba["crash_prob"],proba["squeeze_prob"],proba["breakout_prob"]]
    colors = [DN_CLR,"#b8860b",UP_CLR]
    fig = go.Figure(go.Bar(x=values,y=labels,orientation="h",
        marker_color=colors,opacity=0.85,
        text=[f"{v:.1%}" for v in values],textposition="outside",
        textfont=dict(size=13,family="Space Mono",color="#2c2c2c")))
    fig.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT_CLR),
        xaxis=dict(range=[0,1],gridcolor=GRID_CLR,tickformat=".0%"),
        yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=20,r=80,t=20,b=20),height=200,showlegend=False)
    return fig


def chart_fi(fi: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=fi.values[::-1],y=fi.index[::-1],orientation="h",
        marker=dict(color=fi.values[::-1],
                    colorscale=[[0,"#d4e8f5"],[0.5,"#3a6ea5"],[1,"#1a3a6e"]],
                    showscale=False),
        text=[f"{v:.2f}" for v in fi.values[::-1]],textposition="outside",
        textfont=dict(size=10,color=TEXT_CLR)))
    fig.update_layout(paper_bgcolor=DARK_BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT_CLR),
        xaxis=dict(gridcolor=GRID_CLR,title="Importance"),
        yaxis=dict(gridcolor=GRID_CLR,tickfont=dict(size=11)),
        margin=dict(l=20,r=80,t=20,b=20),height=520)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    ticker     = st.text_input("Ticker", value="CBRS").upper().strip()
    today      = date.today()
    min_date   = today - timedelta(days=7)
    start_date = st.date_input("Start Date", value=min_date,
                               min_value=date(2010,1,1), max_value=today,
                               help="yfinance 1m: max 7 calendar days back")
    end_date   = st.date_input("End Date", value=today,
                               min_value=start_date,
                               max_value=today+timedelta(days=1))
    st.markdown("---")
    st.markdown("**Model Parameters**")
    horizon     = st.slider("Label Horizon (bars)", 5, 30, 10)
    bt_thresh   = st.slider("Breakout Threshold %", 0.2, 2.0, 0.6, step=0.1)/100
    ct_thresh   = st.slider("Crash Threshold %", -2.0, -0.2, -0.6, step=0.1)/100
    conf_thr    = st.slider("Confidence Gate %", 30, 70, 45)/100
    use_atr_lbl = st.toggle("Dynamic ATR Labels", value=True)
    use_spy_rs  = st.toggle("Relative Strength vs SPY", value=True)
    st.markdown("---")
    st.markdown("**Prophet Settings**")
    use_prophet     = st.toggle("Enable Prophet", value=True,
                                disabled=not PROPHET_AVAILABLE)
    prophet_periods = st.slider("Forecast Horizon (bars)", 10, 120, 60,
                                disabled=not use_prophet)
    prophet_ci      = st.slider("Confidence Interval %", 50, 95, 80,
                                disabled=not use_prophet)/100
    if not PROPHET_AVAILABLE:
        st.caption("⚠️ `pip install prophet` to enable")
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", width='stretch', type="primary")

# ─────────────────────────────────────────────
# TITLE + SUMMARY PANEL
# ─────────────────────────────────────────────
st.markdown("# 🧠 Multi-Timeframe Predictor")
st.markdown("<span style='font-family:Space Mono;font-size:13px;color:#64748b'>"
            "CatBoost · Prophet · 1m/5m/15m/30m · Breakout/Squeeze/Crash</span>",
            unsafe_allow_html=True)
st.markdown("""
<style>
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 8px}
.sb{background:linear-gradient(135deg,#fff,#f9f5ef);border:1px solid #ddd5c8;
    border-radius:10px;padding:14px 16px;box-shadow:0 2px 6px rgba(0,0,0,.05)}
.st{font-family:'Space Mono',monospace;font-size:11px;text-transform:uppercase;
    letter-spacing:1.5px;color:#3a6ea5;margin-bottom:8px}
.si{font-family:Georgia,serif;font-size:12.5px;color:#6b5e52;padding:4px 0;
    border-bottom:1px solid #ede8e0;display:flex;justify-content:space-between}
.si:last-child{border-bottom:none} .si b{color:#2c2c2c;font-weight:600}
.sn{font-family:Georgia,serif;font-size:11.5px;color:#7a6e62;margin-top:14px;
    padding:10px 14px;background:#fff8f0;border-left:3px solid #c8a882;border-radius:0 6px 6px 0}
</style>
<div class="sg">
<div class="sb"><div class="st">📥 Input Data</div>
  <div class="si"><span>Source</span><b>yfinance API</b></div>
  <div class="si"><span>Timeframe</span><b>1m OHLCV</b></div>
  <div class="si"><span>Session</span><b>09:30–16:00 ET</b></div>
  <div class="si"><span>Limit</span><b>≤ 7 calendar days</b></div>
  <div class="si"><span>Extras</span><b>Float · Short Ratio · SPY RS</b></div>
</div>
<div class="sb"><div class="st">⚙️ Features</div>
  <div class="si"><span>1m</span><b>VWAP · ATR · Spread · Candle</b></div>
  <div class="si"><span>5m</span><b>RSI · EMA · Vol Accel</b></div>
  <div class="si"><span>15m</span><b>BB Squeeze · MACD · VWAP Z</b></div>
  <div class="si"><span>30m</span><b>EMA Accel · Vol Trend</b></div>
  <div class="si"><span>New</span><b>Float Util · VPT · Gamma · DTC</b></div>
</div>
<div class="sb"><div class="st">🎯 CatBoost</div>
  <div class="si"><span>Loss</span><b>MultiClass Logloss</b></div>
  <div class="si"><span>Output</span><b>Breakout/Squeeze/Crash %</b></div>
  <div class="si"><span>Labels</span><b>ATR-dynamic thresholds</b></div>
  <div class="si"><span>Reg.</span><b>L2=10 · Bernoulli bootstrap</b></div>
  <div class="si"><span>Gate</span><b>Dynamic by Vol Regime</b></div>
</div>
<div class="sb"><div class="st">📊 Charts</div>
  <div class="si"><span>Price</span><b>Candle + VWAP + EMA9/21</b></div>
  <div class="si"><span>Oscillator</span><b>RSI(14)</b></div>
  <div class="si"><span>Volume</span><b>1m bar + 30m profile</b></div>
  <div class="si"><span>Gap fix</span><b>Integer index (no gaps)</b></div>
  <div class="si"><span>Filter</span><b>Per-day or All</b></div>
</div>
<div class="sb"><div class="st">🔮 Prophet</div>
  <div class="si"><span>Changepoint</span><b>scale=0.8 (high-beta)</b></div>
  <div class="si"><span>Seasonality</span><b>Intraday 390-min cycle</b></div>
  <div class="si"><span>Regressor</span><b>Normalised volume</b></div>
  <div class="si"><span>Output</span><b>Price cone + CI band</b></div>
  <div class="si"><span>Signal</span><b>UP / DOWN / FLAT</b></div>
</div>
<div class="sb"><div class="st">🤝 Agreement</div>
  <div class="si"><span>✅ Agree</span><b>Signal strengthened</b></div>
  <div class="si"><span>⚠️ Disagree</span><b>Exercise caution</b></div>
  <div class="si"><span>⚪ Uncertain</span><b>Defer to Prophet</b></div>
  <div class="si"><span>MarketCap</span><b>Auto-adjusts depth/leaf</b></div>
  <div class="si"><span>Vol Regime</span><b>Dynamic conf. gate</b></div>
</div>
</div>
<div class="sn">⚠️ For research reference only — not financial advice.</div>
""", unsafe_allow_html=True)
st.markdown("---")

if not run_btn:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis**.")
    st.stop()

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
start_str = start_date.strftime("%Y-%m-%d")
end_str   = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_1m(t, s, e):    return fetch_1m(t, s, e)

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_spy(s, e):      return fetch_spy_1m(s, e)

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_info(t):        return get_info(t)

with st.spinner(f"Fetching {ticker} …"):
    df_1m = _fetch_1m(ticker, start_str, end_str)
if df_1m.empty:
    st.error(f"❌ No data for **{ticker}**. Check ticker or date range.")
    st.stop()

df_spy = pd.DataFrame()
if use_spy_rs:
    with st.spinner("Fetching SPY …"):
        df_spy = _fetch_spy(start_str, end_str)

with st.spinner("Fetching fundamental info …"):
    info = _fetch_info(ticker)

mc_tier    = marketcap_tier(info.get("market_cap"))
dates_found = sorted(set(df_1m.index.date))

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.success(f"✅ {len(df_1m):,} bars  |  {[str(d) for d in dates_found]}")
col_info2.info(f"🏢 {mc_tier['label']}")
if info.get("float_shares"):
    col_info3.info(f"🔄 Float: {info['float_shares']/1e6:.1f}M shares")
if info.get("short_percent_float"):
    col_info4.warning(f"⚠️ Short Float: {info['short_percent_float']*100:.1f}%")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
selected_date = "All"
if len(dates_found) > 1:
    selected_date = st.selectbox("📅 Chart Date",
                                 ["All"] + [str(d) for d in dates_found])
df_plot = df_1m if selected_date=="All" else \
    df_1m[df_1m.index.date == date.fromisoformat(selected_date)]

st.plotly_chart(chart_price(df_plot, f"{ticker} — {selected_date} (1m)"),
                width='stretch')
st.plotly_chart(chart_volume_profile(df_1m), width='stretch')
st.markdown("---")

# ─────────────────────────────────────────────
# FEATURE BUILD + TRAIN
# ─────────────────────────────────────────────
with st.spinner("Building feature matrix …"):
    anchor = str(dates_found[0]) if dates_found else None
    master = build_master(df_1m,
                          df_spy=df_spy if use_spy_rs else None,
                          info=info, anchor_date=anchor)

with st.spinner("Training CatBoost …"):
    master["label"] = make_labels(master, horizon=horizon,
                                  bt=bt_thresh, ct=ct_thresh,
                                  use_atr=use_atr_lbl,
                                  atr_mult=mc_tier["atr_mult"])
    labeled    = master.dropna(subset=["label"])
    feat_cols  = get_feature_cols(labeled)
    X = labeled[feat_cols]; y = labeled["label"].astype(int)
    lbl = label_summary(y)

    vol_regime   = master["m1_vol_regime"].iloc[-1] \
                   if "m1_vol_regime" in master.columns else 1
    eff_gate     = dynamic_confidence_gate(conf_thr, vol_regime)
    regime_map   = {0:"🟢 Low Vol",1:"🟡 Mid Vol",2:"🔴 High Burst"}
    regime_lbl   = regime_map.get(int(vol_regime),"—")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔴 Crash",    lbl["crash"])
    c2.metric("🟡 Squeeze",  lbl["squeeze"])
    c3.metric("🟢 Breakout", lbl["breakout"])
    c4.metric("🎯 Gate",     f"{eff_gate:.0%}",
              delta=f"{(eff_gate-conf_thr)*100:+.0f}%", delta_color="off")

    if len(X)<50 or len(y.unique())<2:
        st.warning("⚠️ Insufficient data. Extend date range.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def _train(key, _X, _y, _tier):
        return train(_X, _y, mc_tier=_tier, return_report=True)

    cache_key = f"{ticker}|{start_str}|{end_str}|{use_atr_lbl}|{use_spy_rs}|{hash(str(X.values.tobytes()[:256]))}"
    model, report, val_acc = _train(cache_key, X, y, mc_tier)

with st.expander("📋 Validation Report", expanded=False):
    st.code(report)
    st.metric("Validation Accuracy", f"{val_acc:.2%}")

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
st.markdown("## 🎯 Latest Bar Prediction")
result = model_predict(model, master, feat_cols, eff_gate)
proba  = result
trend  = result["predicted_trend"]
tclass = {"BREAKOUT":"trend-breakout","CRASH":"trend-crash",
          "SQUEEZE":"trend-squeeze"}.get(trend,"trend-uncertain")
valid  = result["signal_valid"]

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, label, val, color in [
    (c1,"Last Close",  f"${result['close']}",    "#1a1a2e"),
    (c2,"Breakout",    f"{result['breakout_prob']:.1%}", "#1a7a3f"),
    (c3,"Squeeze",     f"{result['squeeze_prob']:.1%}",  "#b8860b"),
    (c4,"Crash",       f"{result['crash_prob']:.1%}",    "#c0392b"),
]:
    col.markdown(f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{val}</div>
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

c6.markdown(f"""<div class="metric-card">
  <div class="metric-label">Vol Regime</div>
  <div class="metric-value" style="font-size:16px;margin-top:8px">{regime_lbl}</div>
  <div style="font-family:'Space Mono',monospace;font-size:11px;color:#8a7968;margin-top:4px">
    Gate: {eff_gate:.0%}
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.plotly_chart(chart_proba(result), width='stretch')

# ─────────────────────────────────────────────
# PROPHET
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔮 Prophet Forecast "
            "<span style='font-size:13px;color:#8a7968;font-family:Georgia,serif'>"
            "for reference only</span>", unsafe_allow_html=True)

if not PROPHET_AVAILABLE:
    st.warning("Prophet not installed. Run `pip install prophet`.")
elif not use_prophet:
    st.info("Prophet disabled. Toggle on in sidebar.")
else:
    @st.cache_data(ttl=300, show_spinner=False)
    def _prophet(cache_key, _df, periods, ci):
        return fit_prophet(_df, periods=periods, interval_width=ci)

    with st.spinner("Fitting Prophet …"):
        ph = _prophet(f"{ticker}|{start_str}|{end_str}",
                      regular_session(df_1m),
                      prophet_periods, prophet_ci)

    if ph is None:
        st.warning("⚠️ Not enough data for Prophet (need ≥ 60 bars).")
    else:
        sig_color = {"UP":"#1a7a3f","DOWN":"#c0392b"}.get(
            "UP" if "UP" in ph["prophet_signal"] else
            "DOWN" if "DOWN" in ph["prophet_signal"] else "", "#b8860b")

        pa,pb,pc,pd_ = st.columns(4)
        for col, lbl, val, clr in [
            (pa, "Last Close",       f"${ph['last_actual']:.2f}", "#1a1a2e"),
            (pb, "Next Bar Target",  f"${ph['next_yhat']:.2f}",   "#3a6ea5"),
            (pc, "Expected Δ",       f"{ph['trend_chg_pct']:+.3f}%", sig_color),
            (pd_,"Prophet Signal",   ph["prophet_signal"],         sig_color),
        ]:
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value" style="color:{clr};font-size:{'22px' if lbl=='Prophet Signal' else '26px'}">{val}</div>
            </div>""", unsafe_allow_html=True)

        lo,hi = ph["next_yhat_lo"], ph["next_yhat_hi"]
        st.markdown(f"<div style='font-family:Space Mono;font-size:12px;color:#8a7968;"
                    f"text-align:center;padding:6px 0'>{int(prophet_ci*100)}% CI: "
                    f"<b style='color:#1a1a2e'>${lo:.2f}</b> — "
                    f"<b style='color:#1a1a2e'>${hi:.2f}</b></div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(chart_prophet(ph, ticker), width='stretch')

        agr = model_agreement(trend, ph["prophet_signal"])
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#fff,#f5f0e8);"
            f"border:1.5px solid {agr['color']};border-radius:10px;padding:14px 20px;"
            f"font-family:Georgia,serif;font-size:13px;color:{agr['color']};"
            f"text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06)'>"
            f"CatBoost: <b>{trend}</b> &nbsp;|&nbsp; Prophet: <b>{ph['prophet_signal']}</b>"
            f"<br><span style='font-size:15px;margin-top:6px;display:block'>{agr['message']}</span>"
            f"</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Top-20 Feature Importance")
fi = feature_importance(model, feat_cols, top_n=20)
st.plotly_chart(chart_fi(fi), width='stretch')

st.markdown("<div style='text-align:center;color:#c8bfb4;font-family:Georgia,serif;"
            "font-size:11px;padding:12px'>MTF Predictor · Research only · Not financial advice</div>",
            unsafe_allow_html=True)
