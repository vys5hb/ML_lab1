# %% 
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# %%
# Calculate prevalence
def prev(y):
    return y.mean()

# %%
# One-hot encode categorical variables
def binary_encode(df, col, pos_col):
    df[col] = df[col].apply(lambda x: 1 if x == pos_col else 0)
    
# %%
# Select columns from the dataframe into a new dataframe
def select_cols(df, cols):
    return df[cols].copy()

# %%
# Split your training features and target feature
def split_target(df, target, features):
    y = df[target].copy()
    X = df[features].copy()
    return X, y

# %%
# Partition data into training, tuning, and testing with 60/20/20
def train_tune_test_split(X, y, train_size=.6, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)
    X_tune, X_test, y_tune, y_test = train_test_split(X_temp, y_temp, train_size=.5, random_state=random_state, stratify=y_temp)
    
    return X_train, X_tune, X_test, y_train, y_tune, y_test

# %%
# Standardize continuous variables
def standardize(df, cont_vars):
    df[cont_vars] = StandardScaler().fit_transform(df[cont_vars])
    return df[cont_vars]

# %%
# One-hot encode variables
def one_hot_encode(df, col):
    df = df.copy()
    dummies = pd.get_dummies(df[col], prefix=col+'_', dtype=int)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df

# %%
