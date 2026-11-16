import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

SEED = 42
POOL = True
# The script will test the client's current flat rule against your model
METHODS = ["flat", "xgboost"] 

def fit_methods(train):
    """Train the model on historical data and return it for backtesting."""
    features = ['hotel', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled']
    X = pd.get_dummies(train[features])
    y = train['is_canceled']

    # Train XGBoost with your previous settings
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5, 
        eval_metric='logloss', early_stopping_rounds=20, random_state=SEED
    )
    
    # Isolate a small validation set purely for early stopping during training
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=SEED)
    xgb_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

    return {
        "flat": None, # The baseline rule doesn't need a model object
        "xgboost": {"model": xgb_model, "columns": X.columns}
    }

def predict(method, model_obj, rows):
    """Generate cancellation probabilities for the backtest."""
    if method == "flat":
        # The client's historical average cancellation rate is ~27%
        return np.full(len(rows), 0.27)
    
    if method == "xgboost":
        features = ['hotel', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled']
        X = pd.get_dummies(rows[features])
        
        # Ensure the live columns perfectly match the training columns
        X = X.reindex(columns=model_obj["columns"], fill_value=0)
        
        return model_obj["model"].predict_proba(X)[:, 1]

def recommend(expected_cancellations, buffer):
    """The client's business rule: Expected cancellations minus the safety buffer."""
    return np.maximum(0, np.floor(expected_cancellations - buffer))
