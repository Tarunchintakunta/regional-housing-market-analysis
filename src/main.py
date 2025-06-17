import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Import project modules
from data.preprocess import load_and_sample_data, split_data, preprocess_numerical_features
from data.harmonize import harmonize_king_county, harmonize_madrid, harmonize_australian
from features.build_features import create_price_per_sqm, create_room_features, create_location_features
from models.linear_model import train_linear_model, save_model as save_linear_model
from models.tree_ensemble import train_xgboost, train_lightgbm, save_model as save_tree_model
from models.deep_learning import train_neural_network, save_model as save_dl_model
from models.evaluate import evaluate_model, compare_models
from models.transfer_learning import evaluate_transfer_learning

def process_dataset(region, raw_data_path, processed_data_dir):
    """Process a single dataset"""
    print(f"Processing {region} dataset...")
    
    # Load data
    if region == 'US_PacificNW':
        df = load_and_sample_data(raw_data_path)
        df_harmonized = harmonize_king_county(df)
    elif region == 'EU_Madrid':
        df = load_and_sample_data(raw_data_path)
        df_harmonized = harmonize_madrid(df)
    elif region == 'AU_Housing':
        df = load_and_sample_data(raw_data_path)
        df_harmonized = harmonize_australian(df)
    else:
        raise ValueError(f"Unknown region: {region}")
    
    # Feature engineering
    df_features = create_price_per_sqm(df_harmonized)
    df_features = create_room_features(df_features)
    df_features = create_location_features(df_features, region)
    
    # Save processed data
    processed_file = os.path.join(processed_data_dir, f"{region}_processed.csv")
    df_features.to_csv(processed_file, index=False)
    print(f"Saved processed data to {processed_file}")
    
    return df_features

def train_models(region, df, models_dir):
    """Train all models on a specific region"""
    print(f"Training models for {region}...")
    
    # Prepare data for modeling
    y = df['price_usd']
    X = df.drop(['price_usd', 'region'], axis=1)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Preprocess features
    X_train_num, X_val_num, X_test_num, _, _ = preprocess_numerical_features(X_train, X_val, X_test)
    
    # Train linear model
    print("Training linear model...")
    linear_results = train_linear_model(X_train_num, y_train, X_val_num, y_val, model_type='ridge', alpha=1.0)
    linear_model = linear_results['model']
    
    # Train tree ensemble
    print("Training XGBoost model...")
    xgb_results = train_xgboost(X_train_num, y_train, X_val_num, y_val)
    xgb_model = xgb_results['model']
    
    # Train neural network
    print("Training neural network...")
    nn_results = train_neural_network(X_train_num, y_train, X_val_num, y_val)
    nn_model = nn_results['model']
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    linear_path = os.path.join(models_dir, f"{region}_linear_{timestamp}.joblib")
    save_linear_model(linear_model, linear_path)
    
    xgb_path = os.path.join(models_dir, f"{region}_xgboost_{timestamp}.joblib")
    save_tree_model(xgb_model, xgb_path)
    
    nn_path = os.path.join(models_dir, f"{region}_nn_{timestamp}")
    save_dl_model(nn_model, nn_path)
    
    # Evaluate on test set
    linear_eval, _ = evaluate_model(linear_model, X_test_num, y_test, "Linear Regression", region)
    xgb_eval, _ = evaluate_model(xgb_model, X_test_num, y_test, "XGBoost", region)
    nn_eval, _ = evaluate_model(nn_model, X_test_num, y_test, "Neural Network", region)
    
    # Return models and test data for transfer learning
    models = {
        'linear': {'model': linear_model, 'eval': linear_eval, 'path': linear_path},
        'xgboost': {'model': xgb_model, 'eval': xgb_eval, 'path': xgb_path},
        'nn': {'model': nn_model, 'eval': nn_eval, 'path': nn_path}
    }
    
    test_data = {
        'X_test': X_test_num,
        'y_test': y_test
    }
    
    return models, test_data

def main():
    parser = argparse.ArgumentParser(description='Regional Housing Market Analysis')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.join(args.data_dir, 'processed'), exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Define regions and file paths
    regions = {
        'US_PacificNW': os.path.join(args.data_dir, 'raw', 'king_county_house_sales.csv'),
        'EU_Madrid': os.path.join(args.data_dir, 'raw', 'madrid_real_estate.csv'),
        'AU_Housing': os.path.join(args.data_dir, 'raw', 'australian_housing.csv')
    }
    
    # Process each dataset
    processed_dfs = {}
    for region, filepath in regions.items():
        processed_dfs[region] = process_dataset(region, filepath, os.path.join(args.data_dir, 'processed'))
    
    # Train models for each region
    region_models = {}
    region_test_data = {}
    for region, df in processed_dfs.items():
        region_models[region], region_test_data[region] = train_models(region, df, args.models_dir)
    
    # Evaluate transfer learning
    transfer_results = []
    for source_region, source_data in region_models.items():
        for target_region, target_data in region_test_data.items():
            if source_region != target_region:
                print(f"Evaluating transfer learning from {source_region} to {target_region}...")
                for model_type, model_info in source_data.items():
                    result = evaluate_transfer_learning(
                        model_info['model'],
                        target_data['X_test'],
                        target_data['y_test'],
                        source_region,
                        target_region
                    )
                    result['model_type'] = model_type
                    transfer_results.append(result)
    
    # Save transfer learning results
    transfer_df = pd.DataFrame(transfer_results)
    transfer_df.to_csv(os.path.join(args.models_dir, 'transfer_learning_results.csv'), index=False)
    print(f"Saved transfer learning results to {os.path.join(args.models_dir, 'transfer_learning_results.csv')}")
    
    print("Done!")

if __name__ == "__main__":
    main()