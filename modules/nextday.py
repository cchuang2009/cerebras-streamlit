"""
mtf_modules/nextday.py
────────────────────────────────────────────
隔日方向標籤 + Walk-Forward 驗證。

Usage
-----
from mtf_modules.nextday import make_nextday_labels, walk_forward
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.utils.class_weight import compute_class_weight


# ──────────────────────────────────────────────
# 標籤定義
# ──────────────────────────────────────────────

LABEL_NAMES = {
    0: "STRONG_DOWN",
    1: "WEAK_DOWN",
    2: "NEUTRAL",
    3: "WEAK_UP",
    4: "STRONG_UP",
}

LABEL_COLORS = {
    0: "#8B0000",   # 深紅
    1: "#c0392b",   # 紅
    2: "#b8860b",   # 金黃
    3: "#1a7a3f",   # 綠
    4: "#004d00",   # 深綠
}


def make_nextday_labels(df_daily: pd.DataFrame,
                         strong_thresh: float = 3.0,
                         weak_thresh:   float = 1.0) -> pd.Series:
    """
    5 類隔日方向標籤。

    用當日 OHLCV → 預測隔日收盤報酬率方向。

    strong_thresh : ±% 以上為強訊號（高 Beta 建議 3%）
    weak_thresh   : ±% 以上為弱訊號（建議 1%）
    """
    next_ret = df_daily["close"].pct_change().shift(-1) * 100

    labels = pd.Series(2, index=df_daily.index, name="label")   # 預設 NEUTRAL
    labels[next_ret >  strong_thresh] = 4   # STRONG_UP
    labels[next_ret >  weak_thresh]   = 3   # WEAK_UP
    labels[next_ret < -strong_thresh] = 0   # STRONG_DOWN
    labels[next_ret < -weak_thresh]   = 1   # WEAK_DOWN

    # 強訊號優先（已由上到下覆蓋，但 strong 要蓋掉 weak）
    labels[next_ret >  strong_thresh] = 4
    labels[next_ret < -strong_thresh] = 0

    return labels


def simplify_labels(labels: pd.Series) -> pd.Series:
    """
    把 5 類簡化為 3 類（資料量不足時使用）：
    0=DOWN (原 0+1), 1=NEUTRAL (原 2), 2=UP (原 3+4)
    """
    mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    return labels.map(mapping).rename("label_3class")


# ──────────────────────────────────────────────
# 模型訓練（隔日預測專用）
# ──────────────────────────────────────────────

def build_nextday_model(n_classes: int = 3) -> CatBoostClassifier:
    """
    CatBoost 隔日預測模型。
    日線資料樣本少 → 更淺的樹 + 更強的正則。
    """
    return CatBoostClassifier(
        depth                 = 4,
        learning_rate         = 0.02,
        iterations            = 500,
        loss_function         = "MultiClass",
        eval_metric           = "Accuracy",
        classes_count         = n_classes,
        l2_leaf_reg           = 15,
        min_data_in_leaf      = 5,
        random_strength       = 2.0,
        bootstrap_type        = "Bernoulli",
        subsample             = 0.8,
        colsample_bylevel     = 0.8,
        early_stopping_rounds = 30,
        random_seed           = 42,
        verbose               = 0,
        thread_count          = -1,
    )


def train_nextday(X: pd.DataFrame,
                  y: pd.Series,
                  n_classes: int = 3,
                  val_size:  int = 5) -> tuple:
    """
    訓練隔日方向模型。

    Parameters
    ----------
    X        : 特徵 DataFrame（日線）
    y        : 標籤 Series（0/1/2 或 0-4）
    n_classes: 類別數（建議用 3 類，樣本少時更穩定）
    val_size : 驗證集最後 N 天

    Returns
    -------
    (model, val_accuracy, val_predictions)
    """
    if len(X) < 10:
        raise ValueError("日線資料不足 10 天，無法訓練")

    n_val   = min(val_size, len(X) // 5)
    X_tr    = X.iloc[:-n_val]
    X_val   = X.iloc[-n_val:]
    y_tr    = y.iloc[:-n_val]
    y_val   = y.iloc[-n_val:]

    if len(y_tr.unique()) < 2:
        raise ValueError("訓練集只有一個類別，請增加資料量")

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sw      = y_tr.map(dict(zip(classes.tolist(), weights.tolist()))).values

    model = build_nextday_model(n_classes)
    model.fit(
        Pool(X_tr, y_tr, weight=sw),
        eval_set       = Pool(X_val, y_val),
        use_best_model = True,
    )

    y_pred   = model.predict(X_val).flatten().astype(int)
    val_acc  = (y_pred == y_val.values).mean()

    return model, val_acc, y_pred


# ──────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────

def walk_forward(X: pd.DataFrame,
                 y: pd.Series,
                 n_classes:  int = 3,
                 min_train:  int = 10,
                 step:       int = 1) -> pd.DataFrame:
    """
    時間序列 Walk-Forward 驗證。
    每次用前 N 天訓練，預測第 N+1 天，滾動前進。

    Parameters
    ----------
    X         : 特徵 DataFrame
    y         : 標籤 Series
    n_classes : 類別數
    min_train : 最少訓練樣本數
    step      : 每次前進幾天

    Returns
    -------
    DataFrame with columns:
        date, actual, predicted, correct, proba_0..N, confidence
    """
    results = []
    n = len(X)

    for end in range(min_train, n - step + 1, step):
        X_tr = X.iloc[:end]
        y_tr = y.iloc[:end]
        X_te = X.iloc[end:end + step]
        y_te = y.iloc[end:end + step]

        if len(y_tr.unique()) < 2:
            continue

        try:
            model = build_nextday_model(n_classes)
            classes = np.unique(y_tr)
            weights = compute_class_weight("balanced",
                                            classes=classes, y=y_tr)
            sw = y_tr.map(dict(zip(classes.tolist(), weights.tolist()))).values
            model.fit(Pool(X_tr, y_tr, weight=sw), verbose=0)

            for i in range(len(X_te)):
                row_X   = X_te.iloc[[i]]
                actual  = int(y_te.iloc[i])
                proba   = model.predict_proba(row_X)[0]
                pred    = int(np.argmax(proba))
                conf    = float(proba.max())

                rec = {
                    "date"      : X_te.index[i],
                    "actual"    : actual,
                    "predicted" : pred,
                    "correct"   : pred == actual,
                    "confidence": round(conf, 4),
                }
                for j, p in enumerate(proba):
                    rec[f"proba_{j}"] = round(float(p), 4)
                results.append(rec)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["actual_name"]    = df["actual"].map(
        {0:"DOWN",1:"NEUTRAL",2:"UP"} if n_classes==3
        else LABEL_NAMES)
    df["predicted_name"] = df["predicted"].map(
        {0:"DOWN",1:"NEUTRAL",2:"UP"} if n_classes==3
        else LABEL_NAMES)
    return df


def walk_forward_summary(wf_df: pd.DataFrame,
                          confidence_gate: float = 0.50) -> dict:
    """
    Walk-Forward 結果摘要。
    """
    if wf_df.empty:
        return {}

    # 全部樣本
    total_acc = wf_df["correct"].mean()

    # 高信心樣本
    high_conf = wf_df[wf_df["confidence"] >= confidence_gate]
    hc_acc    = high_conf["correct"].mean() if len(high_conf) > 0 else 0

    # 方向分佈
    label_acc = wf_df.groupby("actual_name")["correct"].mean().to_dict()

    return {
        "total_samples"    : len(wf_df),
        "total_accuracy"   : round(total_acc, 4),
        "high_conf_samples": len(high_conf),
        "high_conf_accuracy": round(hc_acc, 4),
        "by_class_accuracy": label_acc,
        "avg_confidence"   : round(wf_df["confidence"].mean(), 4),
    }
