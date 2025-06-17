import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_sample_data(filepath, sample_size=20000, random_state=42):
    """Load data and sample to consistent size"""
    df = pd.read_csv(filepath)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    return df

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """Split data into train, validation, and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_numerical_features(X_train, X_val, X_test):
    """Preprocess numerical features"""
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_scaler = StandardScaler()
    
    X_train_num = numeric_imputer.fit_transform(X_train)
    X_train_num = numeric_scaler.fit_transform(X_train_num)
    
    X_val_num = numeric_imputer.transform(X_val)
    X_val_num = numeric_scaler.transform(X_val_num)
    
    X_test_num = numeric_imputer.transform(X_test)
    X_test_num = numeric_scaler.transform(X_test_num)
    
    return X_train_num, X_val_num, X_test_num, numeric_imputer, numeric_scaler