"""
mtf_modules/model.py
────────────────────────────────────────────
CatBoost training, prediction, and evaluation.

Usage
-----
from mtf_modules.model import train, predict, feature_importance
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = ["crash", "squeeze", "breakout"]   # 0 / 1 / 2


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def build_params(n_samples: int,
                 mc_tier:   dict | None = None) -> dict:
    """
    Return CatBoost hyperparameters adapted to sample size and market-cap tier.

    mc_tier : dict from features.marketcap_tier()
              keys: depth, min_leaf
    """
    tier      = mc_tier or {}
    depth     = tier.get("depth",    4 if n_samples < 300 else 6)
    min_leaf  = tier.get("min_leaf", 20 if n_samples < 300 else 10)

    return dict(
        depth                 = depth,
        learning_rate         = 0.01,
        iterations            = 800,
        loss_function         = "MultiClass",
        eval_metric           = "Accuracy",
        classes_count         = 3,
        l2_leaf_reg           = 10,
        min_data_in_leaf      = min_leaf,
        random_strength       = 2.0,
        bootstrap_type        = "Bernoulli",
        subsample             = 0.7,
        colsample_bylevel     = 0.7,
        early_stopping_rounds = 50,
        random_seed           = 42,
        verbose               = 0,
        thread_count          = -1,
    )


def train(X: pd.DataFrame,
          y: pd.Series,
          mc_tier: dict | None = None,
          test_size: float = 0.2,
          return_report: bool = False
         ) -> CatBoostClassifier | tuple:
    """
    Train a CatBoostClassifier with balanced class weights.

    Parameters
    ----------
    X            : feature DataFrame
    y            : integer label Series (0/1/2)
    mc_tier      : from features.marketcap_tier() — adjusts depth/min_leaf
    test_size    : validation split fraction
    return_report: if True, also return (model, report_str, val_accuracy)

    Returns
    -------
    model  (or  (model, report_str, val_accuracy) if return_report=True)
    """
    if len(X) < 50 or len(y.unique()) < 2:
        raise ValueError("Insufficient data: need ≥ 50 samples and ≥ 2 classes.")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Balanced class weights → sample_weight
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sw      = y_tr.map(dict(zip(classes.tolist(), weights.tolist()))).values

    params = build_params(len(X_tr), mc_tier)
    model  = CatBoostClassifier(**params)
    model.fit(
        Pool(X_tr, y_tr, weight=sw),
        eval_set       = Pool(X_val, y_val),
        use_best_model = True,
    )

    if return_report:
        y_pred  = model.predict(X_val).flatten().astype(int)
        report  = classification_report(
            y_val, y_pred, target_names=CLASS_NAMES, zero_division=0
        )
        val_acc = (y_pred == y_val.values).mean()
        return model, report, val_acc

    return model


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────

def predict(model: CatBoostClassifier,
            master: pd.DataFrame,
            feat_cols: list[str],
            confidence_gate: float = 0.45) -> dict:
    """
    Predict trend for the latest bar.

    Returns
    -------
    dict with keys:
        timestamp, close,
        crash_prob, squeeze_prob, breakout_prob,
        confidence, predicted_trend, signal_valid
    """
    last  = master[feat_cols].iloc[[-1]]
    proba = model.predict_proba(last)[0]   # shape (3,)
    max_p = float(proba.max())
    idx   = int(np.argmax(proba))
    trend = CLASS_NAMES[idx] if max_p >= confidence_gate else "uncertain"

    return {
        "timestamp"     : master.index[-1],
        "close"         : round(float(master["close"].iloc[-1]), 4),
        "crash_prob"    : round(float(proba[0]), 4),
        "squeeze_prob"  : round(float(proba[1]), 4),
        "breakout_prob" : round(float(proba[2]), 4),
        "confidence"    : round(max_p, 4),
        "predicted_trend": trend.upper(),
        "signal_valid"  : max_p >= confidence_gate,
    }


# ──────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────

def feature_importance(model: CatBoostClassifier,
                        feat_cols: list[str],
                        top_n: int = 20) -> pd.Series:
    """Return top-N feature importances as a sorted Series."""
    fi = pd.Series(model.get_feature_importance(), index=feat_cols)
    return fi.sort_values(ascending=False).head(top_n)


def walk_forward_score(X: pd.DataFrame,
                       y: pd.Series,
                       n_splits: int = 5,
                       mc_tier:  dict | None = None) -> dict:
    """
    Time-series walk-forward cross-validation.
    Returns mean accuracy and per-fold accuracies.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if len(y_tr.unique()) < 2:
            continue
        try:
            m, _, acc = train(X_tr, y_tr,
                              mc_tier=mc_tier,
                              return_report=True)
            # evaluate on actual val split
            y_pred = m.predict(X_val).flatten().astype(int)
            scores.append((y_pred == y_val.values).mean())
        except Exception:
            pass

    return {
        "fold_scores" : scores,
        "mean_acc"    : round(float(np.mean(scores)), 4) if scores else 0.0,
        "std_acc"     : round(float(np.std(scores)),  4) if scores else 0.0,
    }
