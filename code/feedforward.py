import argparse
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

alphabet_size = 126 - 31 + 1


def chop(v, size):
    r = []
    for i in range(0, len(v) - (size - 1), size):
        r.append(v[i : i + size])
    return r


def encode1(v, c):
    # CR LF = LF
    if c == 10:
        v.append(0)
        return
    if c == 13:
        return

    # tab = space
    if c == 9:
        v.append(1)
        return

    c -= 31
    if c < alphabet_size:
        v.append(c)


def encodes(s):
    if isinstance(s, str):
        s = s.encode()
    v = []
    for c in s:
        encode1(v, c)
    return v


def get_filenames(exts, s):
    r = []
    if os.path.splitext(s)[1] in exts:
        r.append(s)
    for root, dirs, files in os.walk(s):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                r.append(os.path.join(root, file))
    return r


def one_hot(a):
    v = [0.0] * alphabet_size
    v[a] = 1.0
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


def read_chunks(file, size):
    return chop(read_file(file), size)


def read_file(file):
    s = open(file, "rb").read()
    return encodes(s)


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


def tensor(v):
    r = []
    for a in v:
        r.extend(one_hot(a))
    return torch.as_tensor(r)


# unit tests
assert len(encodes("\r")) == 0
assert len(encodes("\n")) == 1
assert encodes("\t") == encodes(" ")
assert encodes("\t") != encodes("a")
assert encodes("~")[0] == alphabet_size - 1

assert chop("abcd", 2) == ["ab", "cd"]
assert chop("abcd", 3) == ["abc"]


exts = set()
exts.add(".java")

# command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", "--scramble", help="amount of scrambling", type=int, default=30
)
parser.add_argument("-s", "--seed", help="random number seed")
parser.add_argument("-z", "--size", help="chunk size", type=int, default=100)
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# options
if args.seed is not None:
    random.seed(options.seed)

# files
files = []
for s in args.files:
    files.extend(get_filenames(exts, s))

# read the data
good = []
for file in files:
    good.extend(read_chunks(file, args.size))
print(f"input {len(good) * args.size} characters")

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
            x = tensor(x)
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

# define the network
hidden_size = 100
epochs = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.size * alphabet_size, hidden_size),
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


def accuracy(ds):
    n = 0
    with torch.no_grad():
        for x, y in ds:
            y = y[0]
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
                "%d\t%f\t%f\t%f" % (epoch, loss, accuracy(train_ds), accuracy(test_ds))
            )
