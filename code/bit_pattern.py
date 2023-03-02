import argparse
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    sys.stderr.write(f"{info.filename}:{info.function}:{info.lineno}: {a}\n")


def make(i):
    d = 0
    v = [0] * args.size
    while i < len(v):
        v[i] = 1
        d += 1
        i += d
    return v


def print_dl(dl):
    for x, y in dl:
        print("x:")
        print(x)
        print(x.shape)
        print(x.dtype)
        print()

        print("y:")
        print(y)
        print(y.shape)
        print(y.dtype)
        break


def scramble(v, n):
    v = v.copy()
    for i in range(n):
        j = random.randrange(len(v))
        k = random.randrange(len(v))
        v[j], v[k] = v[k], v[j]
    return v


def split_train_test(v):
    i = len(v) * 80 // 100
    return v[:i], v[i:]


# command line
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--count", help="number of samples", type=int, default=20)
parser.add_argument(
    "-r", "--scramble", help="amount of scrambling", type=int, default=30
)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("-z", "--size", help="array size", type=int, default=100)
args = parser.parse_args()

# options
if args.seed is not None:
    random.seed(options.seed)
    torch.manual_seed(options.seed)

# generate the data
good = []
for i in range(args.count):
    good.append(make(random.randrange(args.size)))

# prepare the data
bad = [scramble(v, args.scramble) for v in good]

train_good, test_good = split_train_test(good)
train_bad, test_bad = split_train_test(bad)

train_d = []
train_d.extend([(v, 1) for v in train_good])
train_d.extend([(v, 0) for v in train_bad])

test_d = []
test_d.extend([(v, 1) for v in test_good])
test_d.extend([(v, 0) for v in test_bad])


class Dataset1(Dataset):
    def __getitem__(self, i):
        return self.v[i]

    def __init__(self, v):
        self.v = []
        for x, y in v:
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor([y], dtype=torch.float32)
            self.v.append((x, y))

    def __len__(self):
        return len(self.v)


train_ds = Dataset1(train_d)
test_ds = Dataset1(test_d)

batch_size = 8

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

print_dl(train_dl)

# define the network
hidden_size = 100
epochs = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.size, hidden_size),
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


# train the network
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
                "%d\t%f\t%f\t%f"
                % (epoch, loss, accuracy(model, train_ds), accuracy(model, test_ds))
            )
