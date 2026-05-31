# cerebras-streamlit
Analysis of the Cerebras, python, streamlit, prophet

# Summary Panel

| 卡片 | 說明 |
|------|------|
| 📥 Input Data | 資料來源、時段、日期選擇方式 |
| ⚙️ Multi-Scale Features | 1m / 5m / 15m / 30m 各尺度特徵清單 |
| 🎯 CatBoost Model | 演算法、Loss、輸出類型、Confidence Gate |
| 📊 Charts | K 線圖、RSI、Volume bar、30m Profile |
| 🔮 Prophet Forecast | 模型設計、季節性、信賴區間錐形 |
| 🤝 Model Agreement | 兩模型一致／不一致／不確定的解讀邏輯 |

v5, non-trading interval onitted
---
| 問題位置 | 原本 | 修正後 |
|---------|------|--------|
| `make_prophet_chart` X 軸 | `ds_et`（真實時間戳） | 整數序號 `[0…n_hist+n_fut]` |
| Forecast cone | `pd.concat` 時間 index | `idx_fut + idx_fut[::-1]` 整數 |
| `add_vline` 分隔線 | `.timestamp() * 1000`（毫秒時間戳） | `x = n_hist - 0.5`（整數位置） |
| `make_prophet_components_chart` | `fc_hist["ds_et"]` 時間軸 | 整數序號 + `tickvals/ticktext` |
| Tick labels | 無 | 每 30 bar 顯示 `MM/DD HH:MM` |
| 多日分隔 | 無 | `add_vline` 虛線 + 日期 annotation |

v6. 「Beta」（β）Case Included
---
|改善 Imprvement| 位置 Position |說明 Details|
|---------|------|--------|
|ATR 動態標籤|label_bars()|use_atr=True 時閾值隨 ATR 浮動，高 Beta 股不再被固定 ±0.6% 淹沒|
|Volatility |Regimeadd_highbeta_features()|0=低波 / 1=中波 / 2=高爆，驅動動態 Gate|
|Volume Acceleration|同上|hb_vol_accel + hb_vol_burst，爆發前訊號|
|VWAP Z-Score|同上|拉伸後回歸的量化指標|
|Opening Range|同上|前 15 分鐘高低點 + 突破 / 跌破 flag|
|RS vs SPY|同上|相對強弱，可開關|
|動態 Confidence Gate|dynamic_confidence_gate()|高爆 +10%，低波 -5%|
|高 Beta CatBoost 參數|train_model()|depth↓ lr↓ iter↑ l2↑ subsample 0.7|
|Prophet changepoint|run_prophet()|scale 0.3→0.8，n_changepoints 30，range 0.95|
|Vol Regime 卡片|預測區|第 6 張卡片顯示當前 regime + 實際生效 Gate|
|Sidebar 開關|use_atr_lbl / use_spy_rs|可分別開關 ATR 標籤與 SPY RS|
|Catboost option|bootstrap_type| "Bernoulli" and omit bagging_temperature|

v7. 新增至 cbrs_app.py 的內容
---
|新增位置|內容|
|get_ticker_info()|從 yfinance 抓 Beta、Float、Short Ratio、IPO Date|
|MarketCapget_news_sentiment()|免費新聞情緒評分，偵測 S&P 入指 / 鎖定期解禁關鍵字|
|add_ipo_features_1m()|10 個 IPO 專屬特徵，直接加到 1m master DataFrame|
|build_feature_matrix()|新增 info / sentiment 參數，內部呼叫 add_ipo_features_1m|
|主流程|自動 fetch info + sentiment，顯示 Beta/MarketCap/Float/Short/News 資訊欄，新聞標題 expander|
