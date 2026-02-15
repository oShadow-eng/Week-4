"""
Day 19 Activity: Transformation Practice
Tasks:
1) Load skewed feature
2) Apply log1p, sqrt, and Yeo-Johnson
3) Compare before/after
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

#Task 1: Load skewed feature
df = pd.read_csv("day19_transform.csv")

#Task 2: Apply log1p, sqrt, and Yeo-Johnson
# Log1p transformation
df['log1p_spend'] = np.log1p(df['spend'])

# Sqrt transformation
df['sqrt_spend'] = np.sqrt(df['spend'])

# Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson')
df['yeo_johnson_spend'] = pt.fit_transform(df[['spend']])

# TODO: Apply transforms and compare summary stats
print("Original Spend Summary Stats:")
print(df['spend'].describe())
print("\nLog1p Spend Summary Stats:")
print(df['log1p_spend'].describe())
print("\nSqrt Spend Summary Stats:")
print(df['sqrt_spend'].describe())
print("\nYeo-Johnson Spend Summary Stats:")
print(df['yeo_johnson_spend'].describe())
