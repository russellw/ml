import argparse
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# command line
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)


# read the data
size = 10000
window = 5

text = open("/src/linux.c", "rb").read()


def get(i):
    if i < 0:
        return 0
    return text[i]


data = []
for i in range(size):
    x = []
    for j in range(i - window, i):
        x.append(get(j))
    data.append((x, get(i)))


# prepare the data
random.shuffle(data)


def split_train_test(v):
    i = len(v) * 80 // 100
    return v[:i], v[i:]


train, test = split_train_test(data)


def bits(n):
    s = bin(n)[2:]
    s = "0" * (7 - len(s)) + s
    return [int(c == "1") for c in s]


def tensor_bytes(v):
    r = []
    for b in v:
        r.extend(bits(b))
    return torch.as_tensor(r, dtype=torch.float32)


class Dataset1(Dataset):
    def __getitem__(self, i):
        return self.v[i]

    def __init__(self, v):
        self.v = []
        for x, y in v:
            x = tensor_bytes(x)
            y = torch.as_tensor(bits(y), dtype=torch.float32)
            self.v.append((x, y))

    def __len__(self):
        return len(self.v)


train = Dataset1(train)
test = Dataset1(test)

batch_size = 8

train_dl = DataLoader(train, batch_size=batch_size)
test_dl = DataLoader(test, batch_size=batch_size)

for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

# define the network
hidden_size = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(window * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7),
        )

    def forward(self, x):
        return self.layers(x)


model = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def accuracy(model, ds):
    n = 0
    with torch.no_grad():
        for x, y in ds:
            z = model(x).round()
            if y.equal(z):
                n += 1
    return n / len(ds)


# train the network
epochs = 1000
for epoch in range(epochs):
    for bi, (x, y) in enumerate(train_dl):
        loss = criterion(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs / 20) == 0 and not bi:
            print(
                "%d\t%f\t%f\t%f"
                % (epoch, loss, accuracy(model, train), accuracy(model, test))
            )
