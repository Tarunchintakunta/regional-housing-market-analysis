from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

def train_linear_model(X_train, y_train, X_val, y_val, model_type='ridge', **kwargs):
    """Train a linear model with regularization"""
    if model_type == 'ridge':
        model = Ridge(**kwargs)
    elif model_type == 'lasso':
        model = Lasso(**kwargs)
    elif model_type == 'elasticnet':
        model = ElasticNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    results = {
        'model': model,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }
    
    return results

def save_model(model, filepath):
    """Save model to disk"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load model from disk"""
    return joblib.load(filepath)