"""
Day 20 Activity: Integrated Feature Engineering
Tasks:
1) Load dataset
2) Encode categoricals, scale numerics
3) Add interaction and transformed feature
4) Compare baseline vs engineered features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO: Load data from data/day20_integration.csv
#Task 1
df = pd.read_csv("day20_integration.csv")

#Task 2: Encode categoricals, scale numerics

#Scale numeric features
numeric_features = ['pages_viewed', 'session_minutes', 'basket_value']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

#City clear
def clean_city(df):
    df = df.copy()
    df['city'] = df['city'].str.strip().str.lower()#for example, standardize case and remove extra spaces 
    df['city'] = df['city'].replace({'ny': 'New York', 'sf': 'San Francisco', 'la': 'Los Angeles'})#for example, standardize common city abbreviations
    df['city'] = df['city'].fillna('unknown')#for example, fill missing city values with 'unknown'
    return df

#One-hot encode bins and compare
one_hot_encoded_city = pd.get_dummies(df['city'], prefix='city')
one_hot_encoded_device_type = pd.get_dummies(df['device_type'], prefix='device')

#Task 3: Add interaction and transformed feature
# Interaction feature: pages_viewed * session_minutes
df['pages_session_interaction'] = df['pages_viewed'] * df['session_minutes']
# Transformed feature: log of basket_value
df['log_basket_value'] = np.log1p(df['basket_value'])

#Task 4: Compare baseline vs engineered features
print("Baseline Numeric Features Summary Stats:")
print(df[numeric_features].describe())

print("\nEngineered Features Summary Stats:")
engineered_features = numeric_features + ['pages_session_interaction', 'log_basket_value']
print(df[engineered_features].describe())

print("\nOne-Hot Encoded City Features:")
print(one_hot_encoded_city.head())
print(one_hot_encoded_device_type.head())


# TODO: Build engineered features
# TODO: Compare baseline vs engineered (summary stats or model)
