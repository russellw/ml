# baseline regression algorithm for comparison:
# what's the accuracy if we just always predict the average?
import argparse

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# args
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()

# data
df = pd.read_csv(args.filename)
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

# scale values to 0-1 range
X_ar = preprocessing.MinMaxScaler().fit_transform(X_ar)
y_ar = y_ar.reshape(-1, 1)
y_ar = preprocessing.MinMaxScaler().fit_transform(y_ar)
y_ar = y_ar[:, 0]

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_ar, y_ar, random_state=0, test_size=0.25
)
print(f"training data: {X_train.shape} -> {y_train.shape}")
print(f"testing  data: {X_test.shape} -> {y_test.shape}")
print()

# train
avg = np.mean(y_train)
print("average:", avg)
print()

# test
predicted = [avg] * y_test.shape[0]
errors = abs(predicted - y_test)
print("mean squared  error:", np.mean(np.square(errors)))
print("mean absolute error:", np.mean(errors))
