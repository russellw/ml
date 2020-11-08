import argparse

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# args
parser = argparse.ArgumentParser(description="skopt linear regression")
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

# torch tensors
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

# hyperparameters
optim_names = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "ASGD",
    "RMSprop",
    "Rprop",
    "SGD",
]

# LBFGS needs an extra closure parameter
# SparseAdam does not support dense gradients, please consider Adam instead
optims = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}

in_features = X_train.shape[1]
out_features = 1
epochs = 5000
space = [
    Categorical(optim_names, name="optim"),
    Real(10 ** -4, 0.5, "log-uniform", name="lr"),
]


def hparam(x, name):
    for i in range(len(x)):
        if space[i].name == name:
            return x[i]
    raise ValueError(name)


def optim(x):
    return optims[hparam(x, "optim")]


# evaluate hyperparameters
def f(x):
    print(x, end=" ", flush=True)

    # model
    model = nn.Linear(in_features, out_features)
    criterion = nn.MSELoss()
    optimizer = optim(x)(model.parameters(), lr=hparam(x, "lr"))

    # train
    for epoch in range(1, epochs + 1):
        # forward
        output = model(X_tensor)
        cost = criterion(output, y_tensor)

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    c = cost.item()
    print(f"{c}")
    return c


# search hyperparameters
res = gp_minimize(f, space)
print()
x = res.x

# train once more with best hyperparameters
print(x, end=" ", flush=True)

# model
model = nn.Linear(in_features, out_features)
criterion = nn.MSELoss()
optimizer = optim(x)(model.parameters(), lr=hparam(x, "lr"))

# train
for epoch in range(1, epochs + 1):
    # forward
    output = model(X_tensor)
    cost = criterion(output, y_tensor)

    # backward
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

c = cost.item()
print(f"{c}")

# test
with torch.no_grad():
    X_tensor = torch.from_numpy(X_train)
    predicted = model(X_tensor).detach().numpy()
    errors = abs(predicted - y_test)
    print("mean squared  error:", np.mean(np.square(errors)))
    print("mean absolute error:", np.mean(errors))
