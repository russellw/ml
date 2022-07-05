import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

symbols = ("orange", "lemon", "stone")
size = 5


def rand():
    v = []
    while len(v) < size:
        v.append(random.choice(symbols))
    return v


def good(v):
    return int(v.count("orange") > v.count("lemon"))


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
            w.append(v)
    w = pos + neg
    random.shuffle(w)
    return w


def convert1(a):
    v = [0.0] * len(symbols)
    i = symbols.index(a)
    assert i >= 0
    v[i] = 1.0
    return v


def convert(v):
    x = [convert1(a) for a in v]
    y = good(v)
    x = torch.as_tensor(x)
    y = torch.as_tensor([float(y)])
    return x, y


class Dataset1(Dataset):
    def __init__(self, n):
        self.w = [convert(v) for v in rands(n)]

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


batch_size = 8

train_ds = Dataset1(800)
test_ds = Dataset1(200)

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

for x, y in train_dl:
    print(x)
    print(x.shape)
    print(x.dtype)
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
            # this does not work in this form
            # the shapes do not line up
            nn.Linear(size * len(symbols), hidden_size),
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


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        y = y[0]
        with torch.no_grad():
            z = model(x)[0]
        if (y and z > 0.5) or (not y and z <= 0.5):
            n += 1
    return n / len(ds)


for epoch in range(epochs):
    for bi, (x, y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs / 20) == 0 and not bi:
            print(
                "%f\t%f\t%f"
                % (loss, accuracy(model, train_ds), accuracy(model, test_ds))
            )
