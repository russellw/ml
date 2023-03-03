import argparse
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# command line
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("files", nargs="+")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

# files
files = []
types = set()


def get(file):
    t = os.path.splitext(file)[1]
    if not t:
        return
    files.append(file)
    types.add(t)


for s in args.files:
    if os.path.isdir(s):
        for root, dirs, fs in os.walk(s):
            for f in fs:
                get(os.path.join(root, f))
        continue
    get(s)


def order(v):
    v = sorted(v)
    d = {}
    for i in range(len(v)):
        d[v[i]] = i
    return d


types = order(types)

# print stats
def inc(d, key, val):
    if key not in d:
        d[key] = 0
    d[key] += val


counts = {}
sizes = {}
for file in files:
    t = os.path.splitext(file)[1]
    inc(counts, t, 1)
    inc(sizes, t, os.stat(file).st_size)
for t in types:
    print("%-10s %6d %12d" % (t, counts[t], sizes[t]))
print()

# read the data
size = 100
data = []


def chop(v):
    r = []
    for i in range(0, len(v) - (size - 1), size):
        r.append(v[i : i + size])
    return r


for file in files:
    for v in chop(open(file, "rb").read()):
        data.append((v, types[os.path.splitext(file)[1]]))
print("read %d bytes" % (len(data) * size))

# prepare the data
random.shuffle(data)


def split_train_test(v):
    i = len(v) * 80 // 100
    return v[:i], v[i:]


train, test = split_train_test(data)


def one_hot(b, n):
    v = [0] * n
    v[b] = 1
    return v


def tensor_bytes(v):
    r = []
    for b in v:
        r.extend(one_hot(b, 256))
    return torch.as_tensor(r, dtype=torch.float32)


class Dataset1(Dataset):
    def __getitem__(self, i):
        return self.v[i]

    def __init__(self, v):
        self.v = []
        for x, y in v:
            x = tensor_bytes(x)
            y = torch.as_tensor(one_hot(y, len(types)), dtype=torch.float32)
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
exit(0)
# define the network
hidden_size = 100
epochs = 2000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size * 256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(types)),
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
                % (epoch, loss, accuracy(model, train), accuracy(model, test))
            )
