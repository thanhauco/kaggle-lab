import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CabinSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Cabin format: Deck/Num/Side
        X[['Deck', 'Num', 'Side']] = X['Cabin'].str.split('/', expand=True)
        X['Num'] = pd.to_numeric(X['Num'], errors='coerce')
        return X.drop(columns=['Cabin'])

class TotalSpendCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X[amenities] = X[amenities].fillna(0)
        X['TotalSpend'] = X[amenities].sum(axis=1)
        X['HasSpent'] = (X['TotalSpend'] > 0).astype(int)
        return X

def get_preprocessor():
    numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'Num']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return Pipeline(steps=[
        ('cabin_split', CabinSplitter()),
        ('total_spend', TotalSpendCreator()),
        ('transformer', preprocessor)
    ])
