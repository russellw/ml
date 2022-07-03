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

# data
X_train = np.array([[0.0, 0.0], [0.0, 0.3], [0.3, 0.0], [0.3, 0.3]], dtype=np.float32)
y_train = np.array([[0.0], [0.3], [0.3], [0.6]], dtype=np.float32)

# torch tensors
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

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
# This never quite converges on zero error
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
