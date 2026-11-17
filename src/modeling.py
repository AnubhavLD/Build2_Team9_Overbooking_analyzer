
"""Two methods: the client's flat rule and an XGBoost cancellation model.
Both turn a night's bookings into 'expected cancellations'; recommend() turns that into rooms to oversell."""
import numpy as np
import pandas as pd
import xgboost as xgb

from features import encode

SEED = 42
POOL = True
METHODS = ["flat", "xgboost"]

# Settings tried on the CHOOSE period (never on the proof period).
# flat   : rooms of safety buffer subtracted from (rate x bookings)
# xgboost: how many 'spreads' (standard deviations) below the expected cancellations we dare to sell
SETTINGS = {
    "flat": list(range(0, 41)),
    "xgboost": [round(z, 2) for z in np.arange(0.0, 6.01, 0.25)],
}


def fit_methods(train):
    """Fit every method on the training period only and return them in a dict."""
    train = train.sort_values("arrival_date")
    X = encode(train)
    y = train["is_canceled"].astype(int)

    # Early stopping needs a validation slice: use the LAST 20% of the training period (time-ordered),
    # so the model is never tuned on a random mix of past and future.
    cut = int(len(train) * 0.8)
    X_t, X_v, y_t, y_v = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric="logloss",
        early_stopping_rounds=20, random_state=SEED,
    )
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

    return {
        "flat": {"rate": float(y.mean())},                       # client's average cancellation rate
        "xgboost": {"model": model, "columns": list(X.columns)},
    }


def predict(method, model_obj, rows):
    """Probability that each booking cancels, one value per row, same order as rows."""
    if method == "flat":
        return np.full(len(rows), model_obj["rate"])
    if method == "xgboost":
        X = encode(rows, model_obj["columns"])
        return model_obj["model"].predict_proba(X)[:, 1]
    raise ValueError(f"unknown method {method}")


def recommend(method, nights, setting):
    """Rooms to oversell on each night (whole numbers, zero or more)."""
    if method == "flat":
        k = np.floor(nights["expected"] - setting)
    else:
        k = np.floor(nights["expected"] - setting * nights["spread"])
    return np.maximum(0, k).astype(int).to_numpy()