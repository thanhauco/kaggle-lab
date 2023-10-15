import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EpsilonHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if 'Epsilon' in X.columns:
            # Epsilon is essentially a date or unknown categorical
            # We treat 'Unknown' as a separate category if date parsing fails
            X['Epsilon'] = pd.to_datetime(X['Epsilon'], errors='coerce')
            X['Epsilon_Year'] = X['Epsilon'].dt.year.fillna(-1)
            X['Epsilon_Month'] = X['Epsilon'].dt.month.fillna(-1)
            X = X.drop('Epsilon', axis=1)
        return X

class ImputerAndNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = X.mean()
        return self
        
    def transform(self, X):
        X = X.copy()
        return X.fillna(self.means_)

def balanced_log_loss(y_true, y_pred):
    # Competition metric implementation
    # y_true: class labels, y_pred: probabilities
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    
    # Calculate log loss for each class
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    
    # Clip probabilities
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    log_loss_0 = -np.sum((1 - y_true) * np.log(1 - y_pred)) * w_0
    log_loss_1 = -np.sum(y_true * np.log(y_pred)) * w_1
    
    return (log_loss_0 + log_loss_1) / 2
