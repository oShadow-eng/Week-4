"""
Day 18 Activity: Binning Practice
Tasks:
1) Load age dataset
2) Apply equal-width bins, equal-frequency bins, and domain bins
3) One-hot encode bins and compare
"""

import pandas as pd

#Task 1: Load age dataset
df = pd.read_csv("day18_binning.csv")

#Task 2: Apply equal-width bins, equal-frequency bins, and domain bins
df['age_equal_width'] = pd.cut(df['age'], bins=5)
df['age_equal_frequency'] = pd.qcut(df['age'], q=5)

bins_s = [0, 18, 35, 50, 65, 100]
df['age_domain_bins'] = pd.cut(df['age'], bins=bins_s, right=False)

#Task 3: One-hot encode bins and compare
one_hot_encoded_equal_width = pd.get_dummies(df['age_equal_width'], prefix='age')
one_hot_encoded_equal_frequency = pd.get_dummies(df['age_equal_frequency'], prefix='age')
one_hot_encoded_domain_bins = pd.get_dummies(df['age_domain_bins'], prefix='age')

print(f"One-hot encoded equal-width bins:\n {df['age_equal_width'].value_counts().sort_index()}\n")
print(f"One-hot encoded equal-frequency bins:\n {df['age_equal_frequency'].value_counts().sort_index()}\n")
print(f"One-hot encoded domain bins:\n {df['age_domain_bins'].value_counts().sort_index()}\n")




# TODO: Create domain bins
# TODO: Compare value counts
