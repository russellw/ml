# random forest regression tutorial at:
# https://github.com/WillKoehrsen/Data-Analysis/blob/master/random_forest_explained/Random%20Forest%20Explained.ipynb
import argparse
import os
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
import pydot

# args
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()

# data
df = pd.read_csv(args.filename, header=None)
print(df)
print()

# separate the output column
y_name = df.columns[-1]
y_df = df[y_name]
X_df = df.drop(y_name, axis=1)

# one-hot encode categorical features
X_df = pd.get_dummies(X_df)
print(X_df)
print()

# numpy arrays
X_ar = np.array(X_df, dtype=np.float32)
y_ar = np.array(y_df, dtype=np.float32)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_ar, y_ar, random_state=0, test_size=0.2
)
print(f"training data: {X_train.shape} -> {y_train.shape}")
print(f"testing  data: {X_test.shape} -> {y_test.shape}")
print()

# model
rf = RandomForestRegressor()

# train
print("training")
rf.fit(X_train, y_train)

predicted = rf.predict(X_train)
errors = abs(predicted - y_train)
print("mean squared  error:", np.mean(np.square(errors)))
print("mean absolute error:", np.mean(errors))
print()

# test
print("testing")
predicted = rf.predict(X_test)
errors = abs(predicted - y_test)
print("mean squared  error:", np.mean(np.square(errors)))
print("mean absolute error:", np.mean(errors))
print()

# feature importance
importance = [(k, v) for k, v in zip(X_df.columns, rf.feature_importances_)]
importance.sort(key=lambda x: x[1])
print("feature importance:")
for k, v in importance:
    print(f"{k:20} {v:10.6f}")
