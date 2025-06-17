import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

def train_xgboost(X_train, y_train, X_val, y_val, **kwargs):
    """Train an XGBoost model"""
    model = xgb.XGBRegressor(**kwargs)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    results = {
        'model': model,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'feature_importance': model.feature_importances_
    }
    
    return results

def train_lightgbm(X_train, y_train, X_val, y_val, **kwargs):
    """Train a LightGBM model"""
    # Similar implementation as XGBoost
    pass