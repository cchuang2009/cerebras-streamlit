# cerebras-streamlit
Analysis of the Cerebras, python, streamlit, prophet

# Summaru Panel

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
