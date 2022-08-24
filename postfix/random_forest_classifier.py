# results for 10,000 records, one-hot:
# RandomForestClassifier: 93% accuracy
# MLP                   : 95% accuracy

import argparse
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
x_df = df.drop(y_name, axis=1)

# one-hot encode categorical features
x_df = pd.get_dummies(x_df)
print(x_df)
print()

# numpy arrays
x_ar = np.array(x_df, dtype=np.float32)
y_ar = np.array(y_df, dtype=np.float32)

# split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_ar, y_ar, random_state=0, test_size=0.2
)
print(f"training data: {x_train.shape} -> {y_train.shape}")
print(f"testing  data: {x_test.shape} -> {y_test.shape}")
print()

# model
rf = RandomForestClassifier()

# train
print("training")
rf.fit(x_train, y_train)

predicted = rf.predict(x_train)
errors = abs(predicted - y_train)
print(1 - np.sum(errors) / errors.shape[0])
print()

# test
print("testing")
predicted = rf.predict(x_test)
errors = abs(predicted - y_test)
print(1 - np.sum(errors) / errors.shape[0])
print()

# feature importance
importance = [(k, v) for k, v in zip(x_df.columns, rf.feature_importances_)]
importance.sort(key=lambda x: x[1])
print("feature importance:")
for k, v in importance:
    print(f"{k:20} {v:10.6f}")
