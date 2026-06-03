"""
unified_app.py — High-Beta & IPO Predictor
────────────────────────────────────────────
三分頁合一：
  Tab 1 🔍 Scanner    — 自動掃描近期 IPO 高 Beta 標的
  Tab 2 📈 Intraday   — 盤中 1m 預測（breakout/squeeze/crash）+ Prophet
  Tab 3 🎯 Next-Day   — 隔日全日方向預測 + 技術進出點位

Run:  streamlit run unified_app.py
Deps: pip install streamlit plotly yfinance catboost
      scikit-learn pytz pandas numpy prophet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime
import streamlit as st
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Module imports (from modules/) ──
from modules.scanner      import scan_ipo_candidates, fetch_ticker_profile
from modules.ipo_features import (build_ipo_features, news_sentiment,
                                   anchored_vwap as _avwap_fn)
from modules.nextday      import (make_nextday_labels, simplify_labels,
                                   train_nextday, walk_forward,
                                   walk_forward_summary, LABEL_NAMES,
                                   LABEL_COLORS)
from modules.levels       import calculate_levels, format_levels, LevelEngine

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

ET = pytz.timezone("America/New_York")

# ─────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="High-Beta & IPO Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:Georgia,serif;background:#f8f5f0;color:#2c2c2c}
h1,h2,h3,h4{font-family:Georgia,serif;color:#1a1a2e;font-weight:700}
.main .block-container{background:#f8f5f0;padding-top:1.5rem}
div[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#fffdf9,#f3ede4);
    border-right:1px solid #ddd5c8}
.mc{background:linear-gradient(135deg,#fff,#f5f0e8);border:1px solid #ddd5c8;
    border-radius:12px;padding:14px 18px;text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,.06)}
.ml{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;
    color:#8a7968;font-family:'Space Mono',monospace}
.mv{font-size:22px;font-weight:700;font-family:'Space Mono',monospace;margin-top:4px}
.trend-breakout{color:#1a7a3f} .trend-crash{color:#c0392b}
.trend-squeeze{color:#b8860b}  .trend-uncertain{color:#7a7060}
.badge{display:inline-block;padding:4px 14px;border-radius:999px;
       font-size:12px;font-family:'Space Mono',monospace;font-weight:700}
.badge-up  {background:#dcf5e7;color:#1a7a3f;border:1px solid #1a7a3f}
.badge-dn  {background:#fde8e8;color:#c0392b;border:1px solid #c0392b}
.badge-neu {background:#fef9e7;color:#b8860b;border:1px solid #b8860b}
.badge-unc {background:#f0ede8;color:#8a7968;border:1px solid #c8bfb4}
hr{border-color:#ddd5c8}
</style>
""", unsafe_allow_html=True)

BG   = "#f8f5f0"
GRID = "#e8e2d8"
TEXT = "#4a4035"
UP   = "#1a7a3f"
DN   = "#c0392b"
GOLD = "#b8860b"
BLUE = "#3a6ea5"

# ─────────────────────────────────────────────
# SHARED DATA LAYER
# ─────────────────────────────────────────────

def _clean(df):
    if df.empty: return df
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
def get_1m(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1m",
                     progress=False, auto_adjust=True,
                     multi_level_index=False, prepost=True)
    return _clean(df)

@st.cache_data(ttl=300, show_spinner=False)
def get_daily(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True, multi_level_index=False)
    return _clean(df)

@st.cache_data(ttl=120, show_spinner=False)
def get_spy_1m(start, end):
    df = yf.download("SPY", start=start, end=end, interval="1m",
                     progress=False, auto_adjust=True,
                     multi_level_index=False, prepost=False)
    return _clean(df)

@st.cache_data(ttl=600, show_spinner=False)
def get_info(ticker):
    return fetch_ticker_profile(ticker) or {}

@st.cache_data(ttl=300, show_spinner=False)
def get_sentiment(ticker):
    return news_sentiment(ticker)

@st.cache_data(ttl=600, show_spinner=False)
def run_scan(extra, max_days, min_beta, min_vol):
    return scan_ipo_candidates(
        extra_tickers=extra or None,
        max_ipo_days=max_days, min_beta=min_beta,
        min_avg_volume=min_vol, top_n=15,
    )

# ─────────────────────────────────────────────
# INTRADAY FEATURE ENGINE
# ─────────────────────────────────────────────

def _resample(df, rule):
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df[list(agg)].resample(rule, label="left", closed="left").agg(agg).dropna()

def _add_indicators(df, pfx):
    f = df.copy()
    c,h,l,o,v = f["close"],f["high"],f["low"],f["open"],f["volume"]
    for n in [1,3,5,10]: f[f"{pfx}_ret{n}"] = c.pct_change(n)
    if pfx != "1d":
        tp = (h+l+c)/3; dk = f.index.normalize()
        f[f"{pfx}_vwap"]      = (tp*v).groupby(dk).cumsum()/(v.groupby(dk).cumsum()+1e-9)
        f[f"{pfx}_vwap_dist"] = (c-f[f"{pfx}_vwap"])/(f[f"{pfx}_vwap"]+1e-9)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    f[f"{pfx}_atr"]       = tr.rolling(14,min_periods=1).mean()
    f[f"{pfx}_atr_pct"]   = f[f"{pfx}_atr"]/(c+1e-9)
    f[f"{pfx}_atr_ratio"] = (h-l)/(f[f"{pfx}_atr"]+1e-9)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14,min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14,min_periods=1).mean()
    f[f"{pfx}_rsi"]    = 100-100/(1+gain/(loss+1e-9))
    f[f"{pfx}_rsi_ob"] = (f[f"{pfx}_rsi"]>70).astype(int)
    f[f"{pfx}_rsi_os"] = (f[f"{pfx}_rsi"]<30).astype(int)
    ma20 = c.rolling(20,min_periods=1).mean()
    std20= c.rolling(20,min_periods=1).std().fillna(0)
    bb_up= ma20+2*std20; bb_lo=ma20-2*std20
    f[f"{pfx}_bb_pos"]     = (c-bb_lo)/(bb_up-bb_lo+1e-9)
    f[f"{pfx}_bb_width"]   = (bb_up-bb_lo)/(ma20+1e-9)
    f[f"{pfx}_bb_squeeze"] = (f[f"{pfx}_bb_width"]<f[f"{pfx}_bb_width"].rolling(20,min_periods=1).mean()*0.8).astype(int)
    ema9 = c.ewm(span=9,adjust=False).mean(); ema21=c.ewm(span=21,adjust=False).mean()
    f[f"{pfx}_ema_cross"] = (ema9-ema21)/(c+1e-9)
    f[f"{pfx}_ema_sign"]  = np.sign(f[f"{pfx}_ema_cross"])
    f[f"{pfx}_ema_accel"] = f[f"{pfx}_ema_cross"].diff()
    vol_ma = v.rolling(20,min_periods=1).mean()
    f[f"{pfx}_rel_vol"]   = v/(vol_ma+1e-9)
    f[f"{pfx}_vol_spike"] = (f[f"{pfx}_rel_vol"]>2.5).astype(int)
    f[f"{pfx}_spread"]    = (h-l)/(c+1e-9)
    f[f"{pfx}_imbalance"] = ((c-o)/(h-l+1e-9)).rolling(5,min_periods=1).mean()
    body=(c-o).abs(); rng=h-l+1e-9
    f[f"{pfx}_body_ratio"] = body/rng
    f[f"{pfx}_upper_wick"] = (h-pd.concat([c,o],axis=1).max(axis=1))/rng
    f[f"{pfx}_lower_wick"] = (pd.concat([c,o],axis=1).min(axis=1)-l)/rng
    f[f"{pfx}_is_bull"]    = (c>o).astype(int)
    f[f"{pfx}_volatility"] = c.pct_change().rolling(10,min_periods=1).std()
    return f

def _align_htf(df_1m, df_htf, cols):
    return df_htf[cols].shift(1).reindex(df_1m.index, method="ffill")

def _add_ipo_1m(master, info, sentiment):
    """IPO high-beta features on 1m master."""
    f = master.copy(); c=f["close"]; v=f["volume"]
    ipo_date = info.get("ipo_date")
    if ipo_date:
        anchor = pd.Timestamp(str(ipo_date))
        if f.index.tzinfo is not None and anchor.tzinfo is None:
            anchor = ET.localize(anchor)
        sub = f[f.index >= anchor]
        if not sub.empty:
            tp = (sub["high"]+sub["low"]+sub["close"])/3
            avwap = ((tp*sub["volume"]).cumsum()/(sub["volume"].cumsum()+1e-9)).reindex(f.index)
            f["ipo_avwap"]       = avwap
            f["ipo_vs_avwap"]    = (c-avwap)/(avwap+1e-9)
            f["ipo_above_avwap"] = (c>avwap).astype(int)
        idx_n = f.index.normalize().tz_localize(None) if f.index.tzinfo else f.index.normalize()
        ipo_n = anchor.tz_localize(None) if anchor.tzinfo else anchor
        ipo_days = np.maximum((idx_n - ipo_n).days.values, 0)
        f["ipo_days"]      = ipo_days
        f["ipo_week1"]     = (ipo_days<=5).astype(int)
        f["ipo_week2"]     = ((ipo_days>5)&(ipo_days<=10)).astype(int)
        f["ipo_lock_fear"] = (np.maximum(90-ipo_days,0)<14).astype(int)

    fl = info.get("float_shares")
    if fl and fl > 0:
        dk = f.index.normalize()
        cum_v = v.groupby(dk).cumsum()
        f["ipo_float_util"] = cum_v/fl
        f["ipo_float_xtm"]  = (f["ipo_float_util"]>0.5).astype(int)
    else:
        f["ipo_float_util"] = 0.0; f["ipo_float_xtm"] = 0

    f["ipo_dollar_vol"] = c*v
    f["ipo_dvol_rel"]   = f["ipo_dollar_vol"]/(f["ipo_dollar_vol"].rolling(20,min_periods=1).mean()+1e-9)
    ret=c.pct_change().fillna(0)
    f["ipo_vpt"]       = (ret*v).cumsum()
    f["ipo_vpt_slope"] = f["ipo_vpt"].diff(5)
    rvol = v/(v.rolling(20,min_periods=1).mean()+1e-9)
    ret5 = c.pct_change(5).clip(lower=0)
    bbw  = f["m1_bb_width"] if "m1_bb_width" in f.columns else pd.Series(0.02, index=f.index)
    f["ipo_squeeze"]   = ((rvol*ret5)/(bbw+1e-9)).rolling(5,min_periods=1).mean()
    bar_r = (f["high"]-f["low"])/(c+1e-9)
    f["ipo_gamma"]     = (rvol/(bar_r+1e-9)).rolling(5,min_periods=1).mean()
    sr = info.get("short_ratio")
    f["ipo_dtc"] = float(sr) if sr else v.rolling(5,min_periods=1).mean()/(v+1e-9)

    # News
    f["ipo_news_score"]   = float(sentiment.get("sentiment_score",0))
    f["ipo_has_idx"]      = int(sentiment.get("has_index_news",False))
    f["ipo_has_lock"]     = int(sentiment.get("has_lockup_news",False))
    sig = sentiment.get("sentiment_signal","NEUTRAL")
    f["ipo_sent_pos"]     = int(sig=="POSITIVE")
    f["ipo_sent_neg"]     = int(sig=="NEGATIVE")

    mc = info.get("market_cap") or 0
    f["ipo_mc_tier"] = (0 if mc<3e8 else 1 if mc<2e9 else 2 if mc<1e10 else 3)

    # Vol regime + OR + SPY RS (from add_highbeta_features)
    vol_1m   = c.pct_change().rolling(20,min_periods=5).std()
    vol_slow = vol_1m.rolling(60,min_periods=10).mean()
    ratio    = vol_1m/(vol_slow+1e-9)
    f["hb_vol_regime"] = pd.cut(ratio,bins=[0,0.8,1.5,np.inf],labels=[0,1,2]).astype(float).fillna(1)
    vma5=v.rolling(5,min_periods=1).mean(); vma20=v.rolling(20,min_periods=1).mean()
    vma60=v.rolling(60,min_periods=1).mean()
    f["hb_vol_accel"] = (vma5/(vma20+1e-9))/(vma20/(vma60+1e-9))
    f["hb_vol_burst"] = (f["hb_vol_accel"]>2.0).astype(int)
    if "m1_vwap" in f.columns:
        vd = (c-f["m1_vwap"])/(c+1e-9)
        f["hb_vwap_zscore"] = vd/(vd.rolling(20,min_periods=5).std()+1e-9)
    f["_date"] = f.index.normalize(); f["_mins"] = f.index.hour*60+f.index.minute-570
    orh = f[f["_mins"]<15].groupby("_date")["high"].max().rename("_orh")
    orl = f[f["_mins"]<15].groupby("_date")["low"].min().rename("_orl")
    f = f.join(orh,on="_date").join(orl,on="_date")
    or_rng = f["_orh"]-f["_orl"]+1e-9
    f["hb_or_pos"]  = (c-f["_orl"])/or_rng
    f["hb_or_bo"]   = (c>f["_orh"]).astype(int)
    f["hb_or_bd"]   = (c<f["_orl"]).astype(int)
    f.drop(columns=["_date","_mins","_orh","_orl"],inplace=True)
    return f

def build_intraday_master(df_1m, info, sentiment):
    ind_1m  = _add_indicators(df_1m,"m1")
    ind_5m  = _add_indicators(_resample(df_1m,"5min"),"m5")
    ind_15m = _add_indicators(_resample(df_1m,"15min"),"m15")
    ind_30m = _add_indicators(_resample(df_1m,"30min"),"m30")
    master  = ind_1m.copy()
    for ind,pfx in [(ind_5m,"m5"),(ind_15m,"m15"),(ind_30m,"m30")]:
        cols = [c for c in ind.columns if c.startswith(pfx)]
        master = pd.concat([master, _align_htf(df_1m,ind,cols)], axis=1)
    mins = np.maximum(master.index.hour*60+master.index.minute-570,0)
    master["time_mins"]    = mins
    master["time_sin"]     = np.sin(2*np.pi*mins/390)
    master["time_cos"]     = np.cos(2*np.pi*mins/390)
    master["is_first_30m"] = (mins<30).astype(int)
    master["is_power_hour"]= (mins>330).astype(int)
    master = _add_ipo_1m(master, info, sentiment)
    master.dropna(inplace=True)
    return master

def label_intraday(df, horizon=10, bt=0.006, ct=-0.006, use_atr=True):
    c  = df["close"]
    fh = c.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fl = c.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    fm = (fh-c)/(c+1e-9); fn = (fl-c)/(c+1e-9)
    if use_atr and "m1_atr" in df.columns:
        atr_n = df["m1_atr"].rolling(14,min_periods=1).mean()/(c+1e-9)
        bt_d  =  atr_n.clip(lower=abs(bt))
        ct_d  = -atr_n.clip(lower=abs(ct))
    else:
        bt_d = pd.Series(bt, index=df.index)
        ct_d = pd.Series(ct, index=df.index)
    lb = pd.Series(1, index=df.index)
    lb[fm>bt_d]=2; lb[fn<ct_d]=0
    both=(fm>bt_d)&(fn<ct_d)
    lb[both&(fm.abs()>=fn.abs())]=2; lb[both&(fm.abs()<fn.abs())]=0
    return lb

def dynamic_gate(base, master):
    if "hb_vol_regime" not in master.columns: return base
    r = master["hb_vol_regime"].iloc[-1]
    if r>=2: return min(base+0.10,0.75)
    elif r>=1: return base
    else: return max(base-0.05,0.30)

@st.cache_resource(show_spinner=False)
def train_intraday(key, _X, _y):
    X,y = _X,_y
    if len(X)<50 or len(y.unique())<2: return None
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.2,shuffle=False)
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced",classes=classes,y=y_tr)
    sw = y_tr.map(dict(zip(classes.tolist(),weights.tolist()))).values
    n = len(X_tr)
    model = CatBoostClassifier(
        depth=4 if n<300 else 6, learning_rate=0.01, iterations=800,
        loss_function="MultiClass", eval_metric="Accuracy", classes_count=3,
        l2_leaf_reg=10, min_data_in_leaf=20 if n<300 else 10,
        random_strength=2.0, bootstrap_type="Bernoulli",
        subsample=0.7, colsample_bylevel=0.7,
        early_stopping_rounds=50, random_seed=42, verbose=0, thread_count=-1,
    )
    model.fit(Pool(X_tr,y_tr,weight=sw), eval_set=Pool(X_val,y_val), use_best_model=True)
    return model

# ─────────────────────────────────────────────
# PROPHET
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def run_prophet(cache_key, _df, periods, ci):
    if not PROPHET_OK: return None
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    reg = _df[(_df.index.time>=pd.Timestamp("09:30").time())&
              (_df.index.time<=pd.Timestamp("16:00").time())].copy()
    if len(reg)<60: return None
    vol_n = reg["volume"].values/(reg["volume"].mean()+1e-9)
    pdf = pd.DataFrame({"ds":reg.index.tz_localize(None),"y":reg["close"].values,"volume_norm":vol_n}).dropna()
    m = Prophet(changepoint_prior_scale=0.8,changepoint_range=0.95,
                seasonality_prior_scale=0.05,interval_width=ci,
                daily_seasonality=False,weekly_seasonality=False,
                yearly_seasonality=False,n_changepoints=30)
    m.add_seasonality(name="intraday",period=390/(60*24),fourier_order=8)
    m.add_regressor("volume_norm",standardize=True)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=periods,freq="1min")
    future["volume_norm"] = float(vol_n[-20:].mean())
    fc = m.predict(future)
    n_h = len(pdf)
    fch = fc.iloc[:n_h].copy(); fcf = fc.iloc[n_h:].copy()
    fch["ds_et"] = pd.to_datetime(fch["ds"]).dt.tz_localize(ET)
    fcf["ds_et"] = pd.to_datetime(fcf["ds"]).dt.tz_localize(ET)
    last = float(pdf["y"].iloc[-1])
    ny = float(fcf["yhat"].iloc[0]) if len(fcf) else last
    nlo= float(fcf["yhat_lower"].iloc[0]) if len(fcf) else last
    nhi= float(fcf["yhat_upper"].iloc[0]) if len(fcf) else last
    chg = (ny-last)/(last+1e-9)
    sig = "📈 UP" if chg>0.003 else "📉 DOWN" if chg<-0.003 else "➡️ FLAT"
    return {"fc_hist":fch,"fc_fut":fcf,"actual_df":pdf,
            "last_actual":last,"next_yhat":ny,"next_yhat_lo":nlo,"next_yhat_hi":nhi,
            "trend_chg_pct":chg*100,"prophet_signal":sig,"periods":periods,"interval_width":ci}

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

def _tick(index, step=30):
    n=len(index); tv=list(range(0,n,step))
    return tv,[index[i].strftime("%m/%d %H:%M") for i in tv]

def _bounds(index):
    return [i for i in range(1,len(index)) if index[i].date()!=index[i-1].date()]

def chart_price_1m(df, title):
    reg = df[(df.index.time>=pd.Timestamp("09:30").time())&
             (df.index.time<=pd.Timestamp("16:00").time())].copy()
    if reg.empty: reg=df.copy()
    n=len(reg); idx=list(range(n))
    tv,tl=_tick(reg.index); bounds=_bounds(reg.index)
    colors=[UP if r>=0 else DN for r in reg["close"].pct_change().fillna(0)]
    tp=(reg["high"]+reg["low"]+reg["close"])/3; dk=reg.index.normalize()
    vwap=(tp*reg["volume"]).groupby(dk).cumsum()/(reg["volume"].groupby(dk).cumsum()+1e-9)
    ema9=reg["close"].ewm(span=9,adjust=False).mean()
    ema21=reg["close"].ewm(span=21,adjust=False).mean()
    delta=reg["close"].diff()
    gain=delta.clip(lower=0).rolling(14,min_periods=1).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=1).mean()
    rsi=100-100/(1+gain/(loss+1e-9))
    vol30=reg["volume"].resample("30min").sum().reindex(reg.index,method="ffill").values
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.55,0.25,0.20],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=idx,open=reg["open"].values,high=reg["high"].values,
        low=reg["low"].values,close=reg["close"].values,
        increasing_line_color=UP,decreasing_line_color=DN,
        increasing_fillcolor=UP,decreasing_fillcolor=DN,name="Price",line_width=1),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=vwap.values,name="VWAP",line=dict(color=GOLD,width=1.5,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema9.values,name="EMA9",line=dict(color=BLUE,width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema21.values,name="EMA21",line=dict(color="#7b4fa6",width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=rsi.values,name="RSI",line=dict(color=BLUE,width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_dash="dot",line_color=DN,line_width=0.8,row=2,col=1)
    fig.add_hline(y=30,line_dash="dot",line_color=UP,line_width=0.8,row=2,col=1)
    fig.add_trace(go.Bar(x=idx,y=reg["volume"].values,marker_color=colors,opacity=0.7,name="Volume"),row=3,col=1)
    fig.add_trace(go.Scatter(x=idx,y=vol30,name="Vol 30m",fill="tozeroy",
        fillcolor="rgba(58,110,165,0.08)",line=dict(color=BLUE,width=1,dash="dot")),row=3,col=1)
    for b in bounds:
        for r in [1,2,3]:
            fig.add_vline(x=b,line_dash="dash",line_color="#a09080",line_width=1.2,row=r,col=1)
    ax=dict(tickvals=tv,ticktext=tl,tickangle=-45,tickfont=dict(size=9,family="Space Mono"),
            gridcolor=GRID,zeroline=False,showspikes=True,spikecolor="#a09080",spikethickness=1)
    fig.update_layout(title=dict(text=title,font=dict(family="Georgia,serif",size=13,color="#1a1a2e")),
        paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        xaxis_rangeslider_visible=False,height=660,
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1,font=dict(size=10)),
        margin=dict(l=60,r=20,t=50,b=55),
        xaxis=dict(**ax),xaxis2=dict(**ax),xaxis3=dict(**ax))
    fig.update_yaxes(gridcolor=GRID,zeroline=False)
    fig.update_yaxes(title_text="Price",row=1,col=1,title_font_size=10)
    fig.update_yaxes(title_text="RSI",row=2,col=1,title_font_size=10,range=[0,100])
    fig.update_yaxes(title_text="Volume",row=3,col=1,title_font_size=10)
    return fig

def chart_vol_profile(df):
    reg = df[(df.index.time>=pd.Timestamp("09:30").time())&
             (df.index.time<=pd.Timestamp("16:00").time())].copy()
    intervals=(reg.index.hour*60+reg.index.minute-570)//30
    reg["interval_30m"]=np.clip(intervals,0,12)
    LABELS={0:"09:30",1:"10:00",2:"10:30",3:"11:00",4:"11:30",5:"12:00",
            6:"12:30",7:"13:00",8:"13:30",9:"14:00",10:"14:30",11:"15:00",12:"15:30"}
    pal=[BLUE,UP,GOLD,"#7b4fa6"]
    fig=go.Figure()
    for i,d in enumerate(sorted(reg.index.normalize().unique())):
        grp=reg[reg.index.normalize()==d].groupby("interval_30m")["volume"].sum().reset_index()
        tot=grp["volume"].sum()
        grp["pct"]=grp["volume"]/(tot+1e-9)*100
        grp["label"]=grp["interval_30m"].map(LABELS)
        fig.add_trace(go.Bar(x=grp["label"],
            y=grp["volume"].apply(lambda v:int(v) if pd.notna(v) else 0),
            name=str(d.date()),marker_color=pal[i%len(pal)],opacity=0.85,
            text=grp["pct"].apply(lambda p:f"{p:.1f}%"),textposition="outside",
            textfont=dict(size=9,color=TEXT)))
    fig.update_layout(title=dict(text="Volume by 30-min Interval",
        font=dict(family="Georgia,serif",size=12,color="#1a1a2e")),
        paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        barmode="group",height=340,
        xaxis=dict(gridcolor=GRID,title="Time (ET)"),yaxis=dict(gridcolor=GRID,title="Volume"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1),
        margin=dict(l=60,r=20,t=45,b=40))
    return fig

def chart_proba_bar(proba_dict):
    labels=["Crash","Squeeze","Breakout"]
    values=[proba_dict["crash"],proba_dict["squeeze"],proba_dict["breakout"]]
    colors=[DN,GOLD,UP]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation="h",marker_color=colors,opacity=0.85,
        text=[f"{v:.1%}" for v in values],textposition="outside",
        textfont=dict(size=12,family="Space Mono",color="#2c2c2c")))
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor="#ffffff",
        font=dict(family="Georgia,serif",color=TEXT),
        xaxis=dict(range=[0,1],gridcolor=GRID,tickformat=".0%"),
        yaxis=dict(gridcolor=GRID),margin=dict(l=20,r=80,t=10,b=10),height=180,showlegend=False)
    return fig

def chart_prophet(result, ticker):
    fch=result["fc_hist"].reset_index(drop=True)
    fcf=result["fc_fut"].reset_index(drop=True)
    act=result["actual_df"].reset_index(drop=True)
    nh=len(act); nf=len(fcf); nt=nh+nf
    ih=list(range(nh)); if_=list(range(nh,nt))
    all_ds=list(act["ds"])+list(fcf["ds"]) if nf else list(act["ds"])
    tv=list(range(0,nt,30))
    tl=[pd.to_datetime(all_ds[i]).strftime("%m/%d %H:%M") if i<len(all_ds) else "" for i in tv]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ih,y=act["y"].values,name="Actual",line=dict(color="#2c2c2c",width=1.5)))
    fig.add_trace(go.Scatter(x=ih,y=fch["yhat"].values,name="Fit",line=dict(color="#7b4fa6",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=ih+ih[::-1],
        y=list(fch["yhat_upper"].values)+list(fch["yhat_lower"].values[::-1]),
        fill="toself",fillcolor="rgba(123,79,166,0.08)",line=dict(color="rgba(0,0,0,0)"),showlegend=False))
    if nf>0:
        fig.add_trace(go.Scatter(x=[nh-1,nh],y=[fch["yhat"].iloc[-1],fcf["yhat"].iloc[0]],
            line=dict(color=BLUE,width=2),showlegend=False))
        fig.add_trace(go.Scatter(x=if_,y=fcf["yhat"].values,name=f"Forecast (+{result['periods']}m)",
            line=dict(color=BLUE,width=2.5)))
        fig.add_trace(go.Scatter(x=if_+if_[::-1],
            y=list(fcf["yhat_upper"].values)+list(fcf["yhat_lower"].values[::-1]),
            fill="toself",fillcolor="rgba(58,110,165,0.10)",line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(result['interval_width']*100)}% CI"))
        fig.add_trace(go.Scatter(x=[nh],y=[result["next_yhat"]],mode="markers+text",
            marker=dict(size=10,color=BLUE,line=dict(color="#f8f5f0",width=2)),
            text=[f"  ${result['next_yhat']:.2f}"],
            textfont=dict(color=BLUE,size=12,family="Space Mono"),
            textposition="middle right",name="Next bar"))
        fig.add_vline(x=nh-0.5,line_dash="dash",line_color="#a09080",line_width=1.2,
            annotation_text=" Forecast →",annotation_font_color="#8a7968",annotation_font_size=10)
    fig.update_layout(title=dict(text=f"{ticker} — Prophet Forecast",
        font=dict(family="Georgia,serif",size=12,color="#1a1a2e")),
        paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        xaxis=dict(tickvals=tv,ticktext=tl,tickangle=-45,tickfont=dict(size=9,family="Space Mono"),
                   gridcolor=GRID,zeroline=False,showspikes=True,spikecolor="#a09080"),
        yaxis=dict(gridcolor=GRID,zeroline=False,title="Price"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1,font=dict(size=10)),
        margin=dict(l=60,r=20,t=50,b=55),height=460,hovermode="x unified")
    return fig

def chart_daily(df, ticker, wf_df=None):
    n=len(df); idx=list(range(n))
    tv=list(range(0,n,max(1,n//8)))
    tl=[df.index[i].strftime("%m/%d") for i in tv]
    colors=[UP if r>=0 else DN for r in df["close"].pct_change().fillna(0)]
    ema5=df["close"].ewm(span=5,adjust=False).mean()
    ema10=df["close"].ewm(span=10,adjust=False).mean()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.70,0.30],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=idx,open=df["open"].values,high=df["high"].values,
        low=df["low"].values,close=df["close"].values,
        increasing_line_color=UP,decreasing_line_color=DN,
        increasing_fillcolor=UP,decreasing_fillcolor=DN,name="Price",line_width=1),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema5.values,name="EMA5",line=dict(color=BLUE,width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=ema10.values,name="EMA10",line=dict(color="#7b4fa6",width=1.2)),row=1,col=1)
    if wf_df is not None and not wf_df.empty:
        for _,row in wf_df.iterrows():
            try: bi=df.index.get_loc(row["date"])
            except: continue
            clr=LABEL_COLORS.get(row["predicted"],GOLD)
            sym="triangle-up" if row["predicted"]>=2 else "triangle-down" if row["predicted"]==0 else "circle"
            fig.add_trace(go.Scatter(x=[bi],y=[df["high"].iloc[bi]*1.01],mode="markers",
                marker=dict(size=9,color=clr,symbol=sym,line=dict(color="white",width=1)),
                showlegend=False,hovertext=f"Pred:{row.get('predicted_name','')} {row['confidence']:.0%}"),row=1,col=1)
    fig.add_trace(go.Bar(x=idx,y=df["volume"].values,marker_color=colors,opacity=0.7,name="Volume"),row=2,col=1)
    ax=dict(tickvals=tv,ticktext=tl,tickangle=-45,tickfont=dict(size=9,family="Space Mono"),gridcolor=GRID,zeroline=False)
    fig.update_layout(title=dict(text=f"{ticker} — Daily Chart",
        font=dict(family="Georgia,serif",size=13,color="#1a1a2e")),
        paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        xaxis_rangeslider_visible=False,height=480,
        legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#ddd5c8",borderwidth=1,font=dict(size=10)),
        margin=dict(l=60,r=20,t=50,b=50),xaxis=dict(**ax),xaxis2=dict(**ax))
    fig.update_yaxes(gridcolor=GRID,zeroline=False)
    fig.update_yaxes(title_text="Price",row=1,col=1,title_font_size=10)
    fig.update_yaxes(title_text="Volume",row=2,col=1,title_font_size=10)
    return fig

def chart_levels(lv):
    if not lv: return go.Figure()
    is_long=lv["is_long"]
    ct=UP if is_long else DN; csl=DN if is_long else UP
    items=[("Target 3",lv["target_3"],ct,"dash"),("Target 2",lv["target_2"],ct,"dot"),
           ("Target 1",lv["target_1"],ct,"solid"),("Entry",lv["entry_ideal"],"#3a6ea5","solid"),
           ("Stop Loss",lv["stop_loss"],csl,"solid"),("Invalidation",lv["invalidation_level"],"#888","dash")]
    fig=go.Figure()
    if is_long:
        fig.add_hrect(y0=lv["entry_ideal"],y1=lv["target_3"],fillcolor="rgba(26,122,63,0.06)",line_width=0)
        fig.add_hrect(y0=lv["stop_loss"],y1=lv["entry_ideal"],fillcolor="rgba(192,57,43,0.06)",line_width=0)
    else:
        fig.add_hrect(y0=lv["target_3"],y1=lv["entry_ideal"],fillcolor="rgba(26,122,63,0.06)",line_width=0)
        fig.add_hrect(y0=lv["entry_ideal"],y1=lv["stop_loss"],fillcolor="rgba(192,57,43,0.06)",line_width=0)
    for name,price,color,dash in items:
        fig.add_hline(y=price,line_color=color,line_dash=dash,line_width=1.5,
            annotation_text=f" {name}: ${price}",annotation_font_color=color,
            annotation_font_size=11,annotation_position="right")
    all_p=[lv["target_3"],lv["target_1"],lv["entry_ideal"],lv["stop_loss"],lv["invalidation_level"],lv["last_close"]]
    fig.add_trace(go.Scatter(x=[0],y=[lv["last_close"]],mode="markers+text",
        marker=dict(size=12,color="#1a1a2e",symbol="diamond",line=dict(color="white",width=2)),
        text=[f"  Close ${lv['last_close']}"],
        textfont=dict(color="#1a1a2e",size=11,family="Space Mono"),textposition="middle right",name="Close"))
    fig.update_layout(title=dict(text=f"{'📈 Long' if is_long else '📉 Short'} — Entry/Exit  "
        f"(RR1:{lv['risk_reward_1']}×  RR2:{lv['risk_reward_2']}×  RR3:{lv['risk_reward_3']}×)",
        font=dict(family="Georgia,serif",size=12,color="#1a1a2e")),
        paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        xaxis=dict(visible=False),
        yaxis=dict(gridcolor=GRID,range=[min(all_p)*0.97,max(all_p)*1.03],tickprefix="$"),
        margin=dict(l=20,r=200,t=50,b=20),height=380,showlegend=False)
    return fig

def chart_wf(wf_df):
    if wf_df.empty: return go.Figure()
    wf=wf_df.copy().reset_index(drop=True)
    wf["cum_acc"]=wf["correct"].expanding().mean()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(wf))),y=wf["cum_acc"].values,
        name="Cumulative Accuracy",line=dict(color=BLUE,width=2),
        fill="tozeroy",fillcolor="rgba(58,110,165,0.08)"))
    fig.add_hline(y=0.5,line_dash="dash",line_color=GOLD,line_width=1,annotation_text="50% random")
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor="#ffffff",font=dict(family="Georgia,serif",color=TEXT),
        xaxis=dict(gridcolor=GRID,title="Trade #"),
        yaxis=dict(gridcolor=GRID,tickformat=".0%",range=[0,1]),
        margin=dict(l=60,r=20,t=30,b=40),height=280)
    return fig

def mc_card(label, value, color="#1a1a2e"):
    return f"""<div class="mc"><div class="ml">{label}</div>
<div class="mv" style="color:{color};font-size:20px">{value}</div></div>"""

def direction_badge(label_idx, confident=True):
    if not confident: return '<span class="badge badge-unc">UNCERTAIN</span>'
    m={4:("badge-up","🚀 STRONG UP"),3:("badge-up","📈 WEAK UP"),
       2:("badge-neu","➡️ NEUTRAL"),1:("badge-dn","📉 WEAK DOWN"),0:("badge-dn","💥 STRONG DOWN")}
    cls,txt=m.get(label_idx,("badge-unc","?"))
    return f'<span class="badge {cls}">{txt}</span>'

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 High-Beta / IPO")
    st.markdown("---")
    ticker = st.text_input("Ticker", value="CBRS").upper().strip()
    today  = date.today()

    st.markdown("**日期設定**")
    start_date = st.date_input("Start Date (1m)",
                               value=today-timedelta(days=6),
                               min_value=date(2010,1,1), max_value=today,
                               help="1m 最多回溯 7 天")
    end_date   = st.date_input("End Date", value=today,
                               min_value=start_date, max_value=today+timedelta(days=1))
    daily_lookback = st.slider("日線回溯天數 (Next-Day)", 30, 365, 60)

    st.markdown("---")
    st.markdown("**掃描設定**")
    max_ipo_days = st.slider("IPO 後天數上限", 7, 180, 90)
    min_beta     = st.slider("最低 Beta", 1.0, 15.0, 3.0, step=0.5)
    extra_input  = st.text_input("額外 Ticker（逗號分隔）", value="CBRS,CRWV")
    extra_tickers= [t.strip().upper() for t in extra_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("**盤中模型**")
    horizon     = st.slider("Label Horizon (bars)", 5, 30, 10)
    bt_thresh   = st.slider("Breakout %", 0.2, 2.0, 0.6, step=0.1)/100
    ct_thresh   = st.slider("Crash %",   -2.0,-0.2,-0.6, step=0.1)/100
    conf_thr    = st.slider("Confidence Gate %", 30, 70, 45)/100
    use_atr_lbl = st.toggle("Dynamic ATR Labels", value=True)
    use_spy     = st.toggle("RS vs SPY", value=True)

    st.markdown("---")
    st.markdown("**Prophet**")
    use_prophet  = st.toggle("Enable Prophet", value=True, disabled=not PROPHET_OK)
    ph_periods   = st.slider("Forecast Bars", 10, 120, 60, disabled=not use_prophet)
    ph_ci        = st.slider("CI %", 50, 95, 80, disabled=not use_prophet)/100

    st.markdown("---")
    st.markdown("**Next-Day 設定**")
    strong_thr   = st.slider("Strong 閾值 %", 1.0, 8.0, 3.0, step=0.5)
    weak_thr     = st.slider("Weak 閾值 %",   0.3, 3.0, 1.0, step=0.1)
    nd_conf      = st.slider("Next-Day Gate %", 30, 80, 50)/100
    atr_sl       = st.slider("止損 ATR", 0.5, 3.0, 1.5, step=0.1)
    atr_tp1      = st.slider("目標一 ATR", 0.5, 3.0, 1.5, step=0.1)
    atr_tp2      = st.slider("目標二 ATR", 1.0, 6.0, 3.0, step=0.5)
    atr_tp3      = st.slider("目標三 ATR", 2.0,10.0, 5.0, step=0.5)

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown("# 🚀 High-Beta & IPO Predictor")
st.markdown(
    "<span style='font-family:Space Mono;font-size:12px;color:#8a7968'>"
    "盤中 1m 預測 · 隔日方向 + 進出點位 · 自動掃描 · 免費資料</span>",
    unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# THREE TABS
# ─────────────────────────────────────────────
tab_scan, tab_intra, tab_nextday = st.tabs([
    "🔍 Scanner",
    "📈 Intraday (1m)",
    "🎯 Next-Day",
])

# ══════════════════════════════════════════════
# TAB 1: SCANNER
# ══════════════════════════════════════════════
with tab_scan:
    st.markdown("### 🔍 IPO High-Beta 候選標的掃描")
    scan_btn = st.button("開始掃描", type="primary")
    if scan_btn:
        with st.spinner("掃描中 …"):
            cands = run_scan(extra_tickers, max_ipo_days, min_beta, 500_000)
        if cands.empty:
            st.warning("無符合條件的標的，請放寬篩選。")
        else:
            dc=[c for c in ["ticker","name","score","beta","ipo_days","current_price",
                             "avg_volume","short_pct_float","market_cap","sector"]
                if c in cands.columns]
            st.dataframe(
                cands[dc].style
                .background_gradient(subset=["score"],cmap="YlGn")
                .format({"score":"{:.0f}","beta":"{:.1f}","ipo_days":"{:.0f}d",
                         "current_price":"${:.2f}","avg_volume":"{:,.0f}",
                         "short_pct_float":"{:.1%}","market_cap":"${:,.0f}"}),
                use_container_width=True, height=380)
            st.info("💡 將上方 Ticker 填入左側欄，切換到 **Intraday** 或 **Next-Day** 分頁分析。")
    else:
        st.info("👈 點擊「開始掃描」自動找出近期 IPO 高 Beta 標的。")

# ══════════════════════════════════════════════
# TAB 2: INTRADAY
# ══════════════════════════════════════════════
with tab_intra:
    st.markdown(f"### 📈 {ticker} — 盤中 1m 預測")
    run_intra = st.button("▶ 執行盤中分析", type="primary", key="run_intra")
    if not run_intra:
        st.info("👈 設定好參數後點擊「執行盤中分析」。")
    else:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str   = (end_date+timedelta(days=1)).strftime("%Y-%m-%d")

        # Fetch
        with st.spinner(f"Fetching {ticker} …"):
            df_1m = get_1m(ticker, start_str, end_str)
        if df_1m.empty:
            st.error("❌ 無資料"); st.stop()

        df_spy = get_spy_1m(start_str,end_str) if use_spy else pd.DataFrame()

        with st.spinner("Fetching fundamentals …"):
            info      = get_info(ticker)
            sentiment = get_sentiment(ticker)

        # Info bar
        beta_v = info.get("beta") or 0
        mc_v   = info.get("market_cap") or 0
        fl_v   = info.get("float_shares") or 0
        sh_v   = info.get("short_pct_float") or 0
        ns_v   = sentiment.get("sentiment_signal","—")
        ns_c   = UP if ns_v=="POSITIVE" else DN if ns_v=="NEGATIVE" else GOLD

        c1,c2,c3,c4,c5 = st.columns(5)
        for col,lbl,val,clr in [
            (c1,"Beta",        f"{beta_v:.1f}","#1a1a2e"),
            (c2,"Market Cap",  f"${mc_v/1e9:.1f}B" if mc_v else "—","#1a1a2e"),
            (c3,"Float",       f"{fl_v/1e6:.1f}M"  if fl_v else "—","#1a1a2e"),
            (c4,"Short Float", f"{sh_v*100:.1f}%"  if sh_v else "—", DN if sh_v>0.2 else "#1a1a2e"),
            (c5,"News",        ns_v, ns_c),
        ]:
            col.markdown(mc_card(lbl,val,clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if sentiment.get("recent_titles"):
            with st.expander("📰 新聞", expanded=False):
                for t in sentiment["recent_titles"]: st.markdown(f"• {t}")

        dates_found = sorted(set(df_1m.index.date))
        sel_date = "All"
        if len(dates_found)>1:
            sel_date = st.selectbox("📅 Chart Date",["All"]+[str(d) for d in dates_found])
        df_plot = df_1m if sel_date=="All" else df_1m[df_1m.index.date==date.fromisoformat(sel_date)]

        st.plotly_chart(chart_price_1m(df_plot,f"{ticker} — {sel_date} (1m)"), use_container_width=True)
        st.plotly_chart(chart_vol_profile(df_1m), use_container_width=True)
        st.markdown("---")

        # Build + Train
        with st.spinner("Building features …"):
            master = build_intraday_master(df_1m, info, sentiment)

        with st.spinner("Training …"):
            master["label"] = label_intraday(master, horizon=horizon,
                                              bt=bt_thresh, ct=ct_thresh, use_atr=use_atr_lbl)
            labeled   = master.dropna(subset=["label"])
            feat_cols = [c for c in labeled.columns
                         if c not in {"open","high","low","close","volume","label"}]
            X = labeled[feat_cols]; y = labeled["label"].astype(int)
            lc = y.value_counts()
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("Crash bars",   int(lc.get(0,0)))
            d2.metric("Squeeze bars", int(lc.get(1,0)))
            d3.metric("Breakout bars",int(lc.get(2,0)))
            vol_r  = master["hb_vol_regime"].iloc[-1] if "hb_vol_regime" in master.columns else 1
            eff_g  = dynamic_gate(conf_thr, master)
            d4.metric("Conf Gate", f"{eff_g:.0%}",
                      delta=f"{(eff_g-conf_thr)*100:+.0f}%", delta_color="off")

            if len(X)<50 or len(y.unique())<2:
                st.warning("⚠️ 資料不足，無法訓練。"); st.stop()

            model = train_intraday(
                f"{ticker}|{start_str}|{end_str}|{use_atr_lbl}|{use_spy}|{hash(str(X.values.tobytes()[:256]))}",
                X, y)

        # Predict
        st.markdown("### 🎯 最新 Bar 預測")
        last = master[feat_cols].iloc[[-1]]
        proba= model.predict_proba(last)[0]
        mx_p = proba.max(); ti = int(np.argmax(proba))
        tmap = {0:"CRASH",1:"SQUEEZE",2:"BREAKOUT"}
        tcls = {"CRASH":"trend-crash","SQUEEZE":"trend-squeeze","BREAKOUT":"trend-breakout"}
        trend= tmap[ti] if mx_p>=eff_g else "UNCERTAIN"
        tclass=tcls.get(trend,"trend-uncertain")
        valid = mx_p>=eff_g
        regime_lbl={0:"🟢 Low",1:"🟡 Mid",2:"🔴 Burst"}.get(int(vol_r),"—")
        proba_d={"crash":proba[0],"squeeze":proba[1],"breakout":proba[2]}

        p1,p2,p3,p4,p5,p6=st.columns(6)
        for col,lbl,val,clr in [
            (p1,"Close",    f"${master['close'].iloc[-1]:.2f}","#1a1a2e"),
            (p2,"Breakout", f"{proba[2]:.1%}",UP),
            (p3,"Squeeze",  f"{proba[1]:.1%}",GOLD),
            (p4,"Crash",    f"{proba[0]:.1%}",DN),
        ]:
            col.markdown(mc_card(lbl,val,clr), unsafe_allow_html=True)

        p5.markdown(f"""<div class="mc">
          <div class="ml">Trend</div>
          <div class="mv {tclass}">{trend}</div>
          <div style="margin-top:5px"><span class="badge {'badge-up' if valid else 'badge-unc'}">
          {'✓ VALID' if valid else '✗ UNCERTAIN'}</span></div>
        </div>""", unsafe_allow_html=True)
        p6.markdown(f"""<div class="mc">
          <div class="ml">Vol Regime</div>
          <div class="mv" style="font-size:16px;margin-top:6px">{regime_lbl}</div>
          <div style="font-family:Space Mono;font-size:10px;color:#8a7968;margin-top:3px">
          Gate:{eff_g:.0%}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(chart_proba_bar(proba_d), use_container_width=True)

        # Prophet
        if use_prophet and PROPHET_OK:
            st.markdown("---")
            st.markdown("### 🔮 Prophet Forecast")
            with st.spinner("Fitting Prophet …"):
                ph = run_prophet(f"{ticker}|{start_str}|{end_str}", df_1m, ph_periods, ph_ci)
            if ph:
                sig_c=UP if "UP" in ph["prophet_signal"] else DN if "DOWN" in ph["prophet_signal"] else GOLD
                q1,q2,q3,q4=st.columns(4)
                for col,lbl,val,clr in [
                    (q1,"Last Close",      f"${ph['last_actual']:.2f}","#1a1a2e"),
                    (q2,"Next Bar Target", f"${ph['next_yhat']:.2f}",BLUE),
                    (q3,"Expected Δ",      f"{ph['trend_chg_pct']:+.3f}%",sig_c),
                    (q4,"Prophet Signal",  ph["prophet_signal"],sig_c),
                ]:
                    col.markdown(mc_card(lbl,val,clr), unsafe_allow_html=True)
                st.plotly_chart(chart_prophet(ph, ticker), use_container_width=True)

                # Agreement
                agree_map={
                    (True, True) :(UP,   "✅ Both AGREE — signal strengthened"),
                    (True, False):(GOLD, "⚠️ Models DISAGREE — exercise caution"),
                    (False,True) :(GOLD, "⚠️ Models DISAGREE — exercise caution"),
                    (False,False):(GOLD, "⚠️ Models DISAGREE — exercise caution"),
                }
                if "UNCERTAIN" in trend:
                    ag_clr,ag_msg="#7a7060","⚪ CatBoost UNCERTAIN — defer to Prophet"
                else:
                    is_cb_up = trend=="BREAKOUT"; is_ph_up="UP" in ph["prophet_signal"]
                    is_cb_dn = trend=="CRASH";    is_ph_dn="DOWN" in ph["prophet_signal"]
                    agree = (is_cb_up and is_ph_up) or (is_cb_dn and is_ph_dn)
                    ag_clr,ag_msg=(UP,"✅ Both AGREE — signal strengthened") if agree else (GOLD,"⚠️ Models DISAGREE")
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#fff,#f5f0e8);"
                    f"border:1.5px solid {ag_clr};border-radius:10px;padding:12px 18px;"
                    f"font-family:Georgia,serif;font-size:13px;color:{ag_clr};text-align:center;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,.06)'>"
                    f"CatBoost: <b>{trend}</b> &nbsp;|&nbsp; Prophet: <b>{ph['prophet_signal']}</b>"
                    f"<br><span style='font-size:14px;margin-top:4px;display:block'>{ag_msg}</span></div>",
                    unsafe_allow_html=True)

        # Feature importance
        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        fi=pd.Series(model.get_feature_importance(),index=feat_cols).sort_values(ascending=False).head(20)
        fig_fi=go.Figure(go.Bar(x=fi.values[::-1],y=fi.index[::-1],orientation="h",
            marker=dict(color=fi.values[::-1],colorscale=[[0,"#d4e8f5"],[0.5,BLUE],[1,"#1a3a6e"]],showscale=False),
            text=[f"{v:.2f}" for v in fi.values[::-1]],textposition="outside",textfont=dict(size=9,color=TEXT)))
        fig_fi.update_layout(paper_bgcolor=BG,plot_bgcolor="#ffffff",
            font=dict(family="Georgia,serif",color=TEXT),
            xaxis=dict(gridcolor=GRID),yaxis=dict(gridcolor=GRID,tickfont=dict(size=9)),
            margin=dict(l=20,r=70,t=10,b=10),height=480)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: NEXT-DAY
# ══════════════════════════════════════════════
with tab_nextday:
    st.markdown(f"### 🎯 {ticker} — 隔日全日方向預測")
    run_nd = st.button("▶ 執行隔日預測", type="primary", key="run_nd")
    if not run_nd:
        st.info("👈 設定好參數後點擊「執行隔日預測」。")
    else:
        today_s   = date.today()
        nd_start  = (today_s-timedelta(days=daily_lookback+5)).strftime("%Y-%m-%d")
        nd_end    = (today_s+timedelta(days=1)).strftime("%Y-%m-%d")
        nd_start1m= (today_s-timedelta(days=7)).strftime("%Y-%m-%d")

        with st.spinner(f"Fetching {ticker} daily …"):
            df_d = get_daily(ticker, nd_start, nd_end)
        with st.spinner(f"Fetching {ticker} 1m (7d) …"):
            df_1m_nd = get_1m(ticker, nd_start1m, nd_end)
        with st.spinner("Fetching fundamentals …"):
            info_nd  = get_info(ticker)
            sent_nd  = get_sentiment(ticker)

        if df_d.empty:
            st.error("❌ 無日線資料"); st.stop()

        # Profile card
        beta_v = info_nd.get("beta") or 0
        mc_v   = info_nd.get("market_cap") or 0
        fl_v   = info_nd.get("float_shares") or 0
        sh_v   = info_nd.get("short_pct_float") or 0
        ipo_d  = info_nd.get("ipo_days") or 0
        ns_v   = sent_nd.get("sentiment_signal","—")
        ns_c   = UP if ns_v=="POSITIVE" else DN if ns_v=="NEGATIVE" else GOLD

        n1,n2,n3,n4,n5,n6=st.columns(6)
        for col,lbl,val,clr in [
            (n1,"Beta",        f"{beta_v:.1f}","#1a1a2e"),
            (n2,"IPO Days",    f"{ipo_d}d","#1a1a2e"),
            (n3,"Market Cap",  f"${mc_v/1e9:.1f}B" if mc_v else "—","#1a1a2e"),
            (n4,"Float",       f"{fl_v/1e6:.1f}M"  if fl_v else "—","#1a1a2e"),
            (n5,"Short Float", f"{sh_v*100:.1f}%"  if sh_v else "—", DN if sh_v>0.2 else "#1a1a2e"),
            (n6,"News",        ns_v, ns_c),
        ]:
            col.markdown(mc_card(lbl,val,clr), unsafe_allow_html=True)

        if sent_nd.get("recent_titles"):
            with st.expander("📰 新聞", expanded=False):
                for t in sent_nd["recent_titles"]: st.markdown(f"• {t}")

        # Features
        with st.spinner("Building features …"):
            feat = build_ipo_features(df_1m_nd, df_d, info_nd, sent_nd)
            feat = feat.reindex(df_d.index, method="ffill").dropna(how="all")
            labels5 = make_nextday_labels(df_d, strong_thresh=strong_thr, weak_thresh=weak_thr)
            labels  = simplify_labels(labels5)
            common  = feat.index.intersection(labels.index)
            feat    = feat.loc[common].dropna()
            labels  = labels.loc[feat.index]
            fc=[c for c in feat.columns if feat[c].dtype in [np.float64,np.int64,float,int] and feat[c].nunique()>1]
            X2=feat[fc]; y2=labels.astype(int)

        lv2=y2.value_counts().rename({0:"DOWN",1:"NEUTRAL",2:"UP"})
        e1,e2,e3,e4=st.columns(4)
        e1.metric("訓練樣本", len(X2))
        e2.metric("📉 DOWN",  int(lv2.get("DOWN",0)))
        e3.metric("➡️ NEUTRAL",int(lv2.get("NEUTRAL",0)))
        e4.metric("📈 UP",    int(lv2.get("UP",0)))

        if len(X2)<10:
            st.warning(f"⚠️ 只有 {len(X2)} 天樣本，至少需要 10 天。請增加「日線回溯天數」。"); st.stop()

        # Walk-Forward
        with st.spinner("Walk-Forward Validation …"):
            wf_df = walk_forward(X2,y2,n_classes=3,min_train=10,step=1)
            wf_sum= walk_forward_summary(wf_df, nd_conf)

        if not wf_df.empty:
            w1,w2,w3,w4=st.columns(4)
            w1.metric("WF 樣本",     wf_sum.get("total_samples",0))
            w2.metric("全體準確率",   f"{wf_sum.get('total_accuracy',0):.1%}")
            w3.metric("高信心樣本",   wf_sum.get("high_conf_samples",0))
            w4.metric("高信心準確率", f"{wf_sum.get('high_conf_accuracy',0):.1%}")
            col_d, col_w = st.columns([3,2])
            with col_d: st.plotly_chart(chart_daily(df_d, ticker, wf_df), use_container_width=True)
            with col_w:
                st.plotly_chart(chart_wf(wf_df), use_container_width=True)
                for cls,acc in wf_sum.get("by_class_accuracy",{}).items():
                    st.markdown(f"`{cls:<8}` {'█'*int(acc*20):<20} **{acc:.1%}**")

        # Train + Predict
        @st.cache_resource(show_spinner=False)
        def _train_nd(key, _X, _y):
            try: return train_nextday(_X,_y,n_classes=3,val_size=3)
            except: return None,0,[]

        nd_key = f"{ticker}|{nd_start}|{daily_lookback}|{strong_thr}|{weak_thr}"
        nd_res = _train_nd(nd_key, X2, y2)
        if nd_res[0] is None:
            st.error("模型訓練失敗，資料可能不足。"); st.stop()

        nd_model, nd_acc, _ = nd_res
        last_f = X2.iloc[[-1]]
        nd_gate  = nd_conf   # float gate value from sidebar
        nd_proba = nd_model.predict_proba(last_f)[0]
        nd_pidx  = int(np.argmax(nd_proba))
        nd_maxp  = float(nd_proba.max())
        nd_valid = nd_maxp >= nd_gate
        map3 = {0:"STRONG_DOWN" if nd_proba[0]>0.55 else "WEAK_DOWN",
                1:"NEUTRAL",
                2:"STRONG_UP"   if nd_proba[2]>0.55 else "WEAK_UP"}
        nd_dir   = map3.get(nd_pidx,"NEUTRAL") if nd_valid else "UNCERTAIN"
        badge_map= {"STRONG_UP":4,"WEAK_UP":3,"NEUTRAL":2,"WEAK_DOWN":1,"STRONG_DOWN":0,"UNCERTAIN":None}

        pred_date = today_s + timedelta(days=1)
        if pred_date.weekday()>=5: pred_date+=timedelta(days=7-pred_date.weekday())

        st.markdown("---")
        st.markdown("### 🎯 明日方向預測")
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#fff,#f5f0e8);"
            f"border:1.5px solid #ddd5c8;border-radius:14px;padding:22px 26px;"
            f"text-align:center;box-shadow:0 2px 10px rgba(0,0,0,.08)'>"
            f"<div style='font-family:Space Mono;font-size:11px;color:#8a7968;"
            f"letter-spacing:1.5px;text-transform:uppercase'>預測日期</div>"
            f"<div style='font-family:Georgia,serif;font-size:20px;color:#1a1a2e;"
            f"font-weight:700;margin:4px 0'>{pred_date.strftime('%Y-%m-%d (%A)')}</div>"
            f"<div style='margin:12px 0'>{direction_badge(badge_map.get(nd_dir),nd_valid)}</div>"
            f"<div style='font-family:Space Mono;font-size:12px;color:#8a7968'>"
            f"Confidence: {nd_maxp:.1%} &nbsp;|&nbsp; "
            f"Val Acc: {nd_acc:.1%} &nbsp;|&nbsp; Gate: {nd_gate:.0%}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability
        nd_fig=go.Figure(go.Bar(
            x=[nd_proba[0],nd_proba[1],nd_proba[2]],y=["DOWN","NEUTRAL","UP"],
            orientation="h",marker_color=[DN,GOLD,UP],opacity=0.85,
            text=[f"{p:.1%}" for p in [nd_proba[0],nd_proba[1],nd_proba[2]]],
            textposition="outside",textfont=dict(size=12,family="Space Mono",color="#2c2c2c")))
        nd_fig.update_layout(paper_bgcolor=BG,plot_bgcolor="#ffffff",
            font=dict(family="Georgia,serif",color=TEXT),
            xaxis=dict(range=[0,1],gridcolor=GRID,tickformat=".0%"),
            yaxis=dict(gridcolor=GRID),margin=dict(l=20,r=80,t=10,b=10),height=180,showlegend=False)
        st.plotly_chart(nd_fig, use_container_width=True)

        # Levels
        st.markdown("---")
        st.markdown("### 📐 建議進出點位")
        avwap_v=None
        if not df_1m_nd.empty and info_nd.get("ipo_date"):
            try:
                av=_avwap_fn(df_1m_nd, str(info_nd["ipo_date"]))
                avwap_v=float(av.dropna().iloc[-1]) if not av.dropna().empty else None
            except: pass

        lv = calculate_levels(df_d, info_nd, nd_dir, nd_proba,
                               atr_mult_sl=atr_sl, atr_mult_tp1=atr_tp1,
                               atr_mult_tp2=atr_tp2, atr_mult_tp3=atr_tp3,
                               avwap_val=avwap_v)

        if lv is None:
            st.info("ℹ️ 方向為 NEUTRAL / UNCERTAIN，不提供點位。")
        else:
            st.plotly_chart(chart_levels(lv), use_container_width=True)
            lc1,lc2=st.columns([3,2])
            with lc1:
                st.markdown("**詳細點位**")
                st.markdown(format_levels(lv))
            with lc2:
                st.markdown("**風報比**")
                st.dataframe(pd.DataFrame({
                    "目標":["T1","T2","T3"],
                    "價格":[f"${lv['target_1']}",f"${lv['target_2']}",f"${lv['target_3']}"],
                    "來源":[lv.get("target_1_source","—"),lv.get("target_2_source","—"),lv.get("target_3_source","—")],
                    "RR":[f"{lv['risk_reward_1']}×",f"{lv['risk_reward_2']}×",f"{lv['risk_reward_3']}×"],
                }), hide_index=True, use_container_width=True)
                if avwap_v:
                    st.markdown(f"<div style='margin-top:8px;padding:8px 12px;background:#f0f8ff;"
                        f"border-left:3px solid {BLUE};font-family:Georgia,serif;font-size:12px;color:#3a5a8a'>"
                        f"AVWAP (IPO)：<b>${avwap_v:.2f}</b></div>", unsafe_allow_html=True)

            with st.expander("🔍 完整技術位明細"):
                lv_df=lv.get("level_df")
                if lv_df is not None and not lv_df.empty:
                    st.dataframe(lv_df.style.background_gradient(subset=["weight"],cmap="YlGn")
                        .format({"price":"${:.2f}","weight":"{:.1f}","vs_close":"{:+.2%}"}),
                        use_container_width=True, height=320)

        # Feature importance
        st.markdown("---")
        st.markdown("### 📊 Feature Importance (Next-Day)")
        fi2=pd.Series(nd_model.get_feature_importance(),index=fc).sort_values(ascending=False).head(15)
        fig_fi2=go.Figure(go.Bar(x=fi2.values[::-1],y=fi2.index[::-1],orientation="h",
            marker=dict(color=fi2.values[::-1],colorscale=[[0,"#d4e8f5"],[0.5,BLUE],[1,"#1a3a6e"]],showscale=False),
            text=[f"{v:.2f}" for v in fi2.values[::-1]],textposition="outside",textfont=dict(size=9,color=TEXT)))
        fig_fi2.update_layout(paper_bgcolor=BG,plot_bgcolor="#ffffff",
            font=dict(family="Georgia,serif",color=TEXT),
            xaxis=dict(gridcolor=GRID),yaxis=dict(gridcolor=GRID,tickfont=dict(size=9)),
            margin=dict(l=20,r=70,t=10,b=10),height=400)
        st.plotly_chart(fig_fi2, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#c8bfb4;font-family:Georgia,serif;"
    "font-size:11px;padding:14px'>High-Beta & IPO Predictor · Research only · Not financial advice</div>",
    unsafe_allow_html=True)
