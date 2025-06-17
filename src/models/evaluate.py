from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name="Model", region="All"):
    """Evaluate model on test data"""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'region': region,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return results, y_pred

def compare_models(evaluation_results):
    """Compare multiple models based on evaluation results"""
    results_df = pd.DataFrame(evaluation_results)
    
    # Create comparison visualizations
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='rmse', hue='region', data=results_df)
    plt.title('RMSE by Model and Region')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='mae', hue='region', data=results_df)
    plt.title('MAE by Model and Region')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='r2', hue='region', data=results_df)
    plt.title('R² by Model and Region')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    return results_df