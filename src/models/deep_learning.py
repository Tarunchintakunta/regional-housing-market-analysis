import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_mlp_model(input_dim, hidden_layers=[64, 32], dropout_rate=0.2):
    """Build a simple MLP model"""
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_neural_network(X_train, y_train, X_val, y_val, **kwargs):
    """Train a neural network model"""
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim, **kwargs)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on validation set
    val_pred = model.predict(X_val).flatten()
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    results = {
        'model': model,
        'history': history,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }
    
    return results

def save_model(model, filepath):
    """Save model to disk"""
    model.save(filepath)

def load_model(filepath):
    """Load model from disk"""
    return tf.keras.models.load_model(filepath)