import argparse
import os
import random
import subprocess
import tempfile

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
for s in args.files:
    if os.path.isdir(s):
        for root, dirs, fs in os.walk(s):
            for f in fs:
                if os.path.splitext(f)[1] == ".exe":
                    files.append(os.path.join(root, f))
        continue
    files.append(s)
print("%d files" % len(files))

# read the data
out_file = os.path.join(tempfile.gettempdir(), "a.asm")

in_size = 100
out_size = 100

data = []
for file in files:
    x = open(file, "rb").read()[:in_size]
    if len(x) < in_size:
        continue

    cmd = "dumpbin", "/disasm", "/nologo", file, "/out:" + out_file
    subprocess.check_call(cmd)
    y = open(out_file, "rb").read()[:in_size]
    if len(y) < out_size:
        continue

    data.append((x, y))
print("%d translations" % len(data))

# prepare the data
random.shuffle(data)


def split_train_test(v):
    i = len(v) * 80 // 100
    return v[:i], v[i:]


train, test = split_train_test(data)


def one_hot(a, n):
    v = [0] * n
    v[a] = 1
    return v


def one_hot_bytes(v):
    r = []
    for b in v:
        r.extend(one_hot(b, 256))
    return r


class Dataset1(Dataset):
    def __getitem__(self, i):
        return self.v[i]

    def __init__(self, v):
        self.v = []
        for x, y in v:
            x = one_hot_bytes(x)
            for i in range(out_size):
                xi = x + one_hot(i, out_size)
                yi = one_hot(y[i], 127)
                self.v.append(
                    (
                        torch.as_tensor(xi, dtype=torch.float32),
                        torch.as_tensor(yi, dtype=torch.float32),
                    )
                )

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
            nn.Linear(in_size * 257, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 127),
        )

    def forward(self, x):
        return self.layers(x)


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        with torch.no_grad():
            z = model(x)
        if torch.argmax(y) == torch.argmax(z):
            n += 1
    return n / len(ds)


# train the network
epochs = 2000
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
