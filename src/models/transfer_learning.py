import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_transfer_learning(source_model, target_X, target_y, source_region, target_region):
    """Evaluate a model trained on source region on target region data"""
    # Predict using source model on target data
    target_pred = source_model.predict(target_X)
    if hasattr(target_pred, 'flatten'):
        target_pred = target_pred.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(target_y, target_pred))
    r2 = r2_score(target_y, target_pred)
    
    results = {
        'source_region': source_region,
        'target_region': target_region,
        'transfer_rmse': rmse,
        'transfer_r2': r2
    }
    
    return results