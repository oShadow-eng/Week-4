"""
Day 17 Activity: Scaling Practice
Tasks:
1) Load numeric dataset
2) Apply Min-Max, Standard, and Robust scaling
3) Compare distributions or summary stats
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


#task 1: Load numeric dataset

data_path = 'day17_scaling.csv'
df = pd.read_csv(data_path)
#task 2: Apply Min-Max, Standard, and Robust scaling
# Assuming the numeric columns are 'feature1', 'feature2', 'feature3'
numeric_cols = ['CRIM', 'RM']
# Min-Max Scaling
min_max_scaler = MinMaxScaler()

df_min_max_scaled = df.copy()
df_min_max_scaled[numeric_cols] = min_max_scaler.fit_transform(df[numeric_cols])
# Standard Scaling
standard_scaler = StandardScaler()
df_standard_scaled = df.copy()

df_standard_scaled[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])
# Robust Scaling
robust_scaler = RobustScaler()
df_robust_scaled = df.copy()
df_robust_scaled[numeric_cols] = robust_scaler.fit_transform(df[numeric_cols])

#task 3: Compare distributions or summary stats
# For demonstration, let's compare the summary statistics of the original and scaled data
print("Original Data Summary:")
print(df[numeric_cols].describe())
print("\nMin-Max Scaled Data Summary:")
print(df_min_max_scaled[numeric_cols].describe())
print("\nStandard Scaled Data Summary:")
print(df_standard_scaled[numeric_cols].describe())



# TODO: Load data from data/day17_scaling.csv
# df = pd.read_csv(...)

# TODO: Fit scalers on numeric columns
# TODO: Transform and compare summaries
