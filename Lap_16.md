"""
Day 16 Activity: Encoding Practice
Tasks:
1) Load categorical dataset
2) Apply label encoding and one-hot encoding
3) Compare model behavior or summary stats
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

#task 1: Load categorical dataset
df = pd.read_csv('day16_encoding.csv')


#task 2: Apply label encoding and one-hot encoding
# Label Encoding
label_encoder = LabelEncoder()
df['city_label_encoded'] = label_encoder.fit_transform(df['city'])

# One-Hot Encoding
one_hot_encoded = pd.get_dummies(df['city'], prefix='city')

#task 3: Compare model behavior or summary stats
# For demonstration, let's compare the unique values in the original, label encoded, and one-hot encoded columns
print("Original 'city' column unique values:")
print(df['city'].unique())
print("\nLabel Encoded 'city' column unique values:")
print(df['city_label_encoded'].unique())
print("\nOne-Hot Encoded 'city' columns:")
print(one_hot_encoded.head())


# TODO: Load data from data/day16_encoding.csv
# df = pd.read_csv(...)

# TODO: Label encode city
# TODO: One-hot encode city
# TODO: Compare outputs
