import random
import statistics

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

size = 5


def good(v):
    return statistics.fmean(v) > 0.5


def rand():
    v = []
    while len(v) < size:
        v.append(random.uniform(0.0, 1.0))
    return v


def rands(n):
    pos = []
    neg = []
    while len(pos) < n / 2 or len(neg) < n / 2:
        v = rand()
        y = good(v)
        if y:
            w = pos
        else:
            w = neg
        if len(w) < n / 2:
            w.append((v, y))
    w = pos + neg
    random.shuffle(w)
    return w


class Dataset1(Dataset):
    def __init__(self, n):
        self.v = rands(n)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]


batch_size = 16

train_dataloader = DataLoader(Dataset1(800), batch_size=batch_size)
test_dataloader = DataLoader(Dataset1(200), batch_size=batch_size)

for X, y in test_dataloader:
    print(f"X: {X}")
    print(f"y: {y.shape} {y.dtype} {y}")
    break

# hyperparameters
hidden_size = 100
epochs = 1000

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train
print("training")
for epoch in range(1, epochs + 1):
    for bi, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if bi % 100 == 0:
            loss, current = loss.item(), bi * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
