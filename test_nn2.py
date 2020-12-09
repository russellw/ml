import torch
import torch.nn as nn

# data
Xs = []
ys = []
n = 5
for i in range(n):
    i1 = i / n
    for j in range(n):
        j1 = j / n
        Xs.append([i1, j1])
        ys.append(i1 + j1)

# torch tensors
X_tensor = torch.tensor(Xs)
y_tensor = torch.tensor(ys)

# https://stackoverflow.com/questions/65219569/pytorch-gives-incorrect-results-due-to-broadcasting
new_shape = (len(ys), 1)
y_tensor = y_tensor.view(new_shape)
print(y_tensor.shape)

# hyperparameters
in_features = len(Xs[0])
hidden_size = 100
out_features = 1
epochs = 500

# model
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.L0 = nn.Linear(in_features, hidden_size)
        self.N0 = nn.ReLU()
        self.L1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.L0(x)
        x = self.N0(x)
        x = self.L1(x)
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
