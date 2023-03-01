import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import zz

exts = set()
exts.add(".java")

# command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", "--scramble", help="amount of scrambling", type=int, default=50
)
parser.add_argument("-s", "--seed", help="random number seed")
parser.add_argument("-z", "--size", help="chunk size", type=int, default=100)
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# options
if args.seed is not None:
    random.seed(options.seed)

# Files
files = []
for s in args.files:
    files.extend(zz.get_filenames(exts, s))

# Read the data
good = []
for file in files:
    good.extend(zz.read_chunks(file, args.size))

# Prepare the data
bad = [zz.scramble(v, args.scramble) for v in good]

train_good, test_good = zz.split_train_test(good)
train_bad, test_bad = zz.split_train_test(bad)

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
            x = zz.tensor(x)
            y = float(y)
            y = torch.as_tensor([y])
            self.v.append((x, y))

    def __len__(self):
        return len(self.v)


train_ds = Dataset1(train_d)
test_ds = Dataset1(test_d)

batch_size = 8

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# Define the network
hidden_size = 100
epochs = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.size * zz.alphabet_size, hidden_size),
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
