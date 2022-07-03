import argparse
import os
import sys

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
parser = argparse.ArgumentParser()
parser.add_argument("-g", action="store_true", help="show graph")
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
y_tensor = torch.from_numpy(y_train).unsqueeze(1)

# hyperparameters
in_features = X_train.shape[1]
hidden_size = 1000
out_features = 1
epochs = 5000

# model
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.L0 = nn.Linear(in_features, hidden_size)
        self.N0 = nn.ReLU()
        self.L1 = nn.Linear(hidden_size, hidden_size)
        self.N1 = nn.Tanh()
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.N2 = nn.ReLU()
        self.L3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.L0(x)
        x = self.N0(x)
        x = self.L1(x)
        x = self.N1(x)
        x = self.L2(x)
        x = self.N2(x)
        x = self.L3(x)
        return x


model = Net(hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train
print("training")
for epoch in range(1, epochs + 1):
    # forward
    output = model(X_tensor)
    cost = criterion(output, y_tensor)

    # backward
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print progress
    if epoch % (epochs // 50) == 0:
        print(f"{epoch:6d} {cost.item():10f}")
print()

# test
with torch.no_grad():
    X_tensor = torch.from_numpy(X_test)
    predicted = model(X_tensor).detach().numpy()
    errors = abs(predicted - y_test)
    print("mean squared  error:", np.mean(np.square(errors)))
    print("mean absolute error:", np.mean(errors))
