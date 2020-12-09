import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# data
df = pd.read_csv("test.csv")
print(df)
print()

# separate the output column
y_name = df.columns[-1]
y_df = df[y_name]
X_df = df.drop(y_name, axis=1)

# numpy arrays
X_ar = np.array(X_df, dtype=np.float32)
y_ar = np.array(y_df, dtype=np.float32)

# torch tensors
X_tensor = torch.from_numpy(X_ar)
y_tensor = torch.from_numpy(y_ar)

# https://stackoverflow.com/questions/65219569/pytorch-gives-incorrect-results-due-to-broadcasting
print(y_tensor.shape)
new_shape = (26, 1)
y_tensor = y_tensor.view(new_shape)
print(y_tensor.shape)

# hyperparameters
in_features = X_ar.shape[1]
hidden_size = 100
out_features = 1
epochs = 500

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


model = Net(hidden_size)
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
    if epoch % (epochs // 10) == 0:
        print(f"{epoch:6d} {cost.item():10f}")
print()

output = model(X_tensor)
cost = criterion(output, y_tensor)
print("mean squared error:", cost.item())
