import pandas as pd
import numpy as np

def create_date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Paydays in Ecuador (15th and last day of month)
    df['is_payday'] = ((df['day'] == 15) | (df['date'].dt.is_month_end)).astype(int)
    return df

def create_lag_features(df, lags=[1, 7, 14, 30]):
    # Ensure data is sorted
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
        
    # Rolling means
    for window in [7, 30]:
        df[f'sales_roll_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    return df

def process_oil_prices(df, oil_path):
    oil = pd.read_csv(oil_path)
    oil['date'] = pd.to_datetime(oil['date'])
    # Fill missing oil prices with interpolation
    oil = oil.set_index('date').resample('D').mean().interpolate().reset_index()
    df = pd.merge(df, oil, on='date', how='left')
    return df
