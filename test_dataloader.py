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
            v = torch.as_tensor(v)
            y = torch.as_tensor([float(y)])
            w.append((v, y))
    w = pos + neg
    random.shuffle(w)
    return w


class Dataset1(Dataset):
    def __init__(self, n):
        self.w = rands(n)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


batch_size = 16

train_dataloader = DataLoader(Dataset1(800), batch_size=batch_size)
test_dataloader = DataLoader(Dataset1(200), batch_size=batch_size)

for X, y in train_dataloader:
    print(X)
    print(X.shape)
    print(X.dtype)
    print(y)
    print(y.shape)
    print(y.dtype)
    break

hidden_size = 100
epochs = 1000


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

for epoch in range(epochs):
    for bi, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs / 20) == 0 and not bi:
            print(f"loss: {loss:>7f}")
