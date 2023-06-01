import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmsle(y_true, y_pred):
    # Clip predictions to be non-negative
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))

def wrmsle(y_true, y_pred, weights):
    # Weighted RMSLE for Store Sales competition
    y_pred = np.maximum(y_pred, 0)
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    return np.sqrt(np.sum(weights * np.square(log_diff)) / np.sum(weights))

def evaluate_model(y_true, y_pred):
    return {
        'RMSLE': rmsle(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
