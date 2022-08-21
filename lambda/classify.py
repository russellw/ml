import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from etc import *
import interpreter
import rand

vocab = ["(", ")", "arg"]
for o, _, _ in interpreter.ops:
    vocab.append(o)

size = 5 * bitLen(len(vocab))
x0s = range(10)


class Dataset1(Dataset):
    def __init__(self, n):
        seen = set()
        pos = []
        neg = []
        while len(pos) < n / 2 or len(neg) < n / 2:
            a = rand.expr(5)
            if a in seen:
                continue
            seen.add(a)
            a = deBruijn(a)

            try:
                if not interpreter.good(a, x0s):
                    continue
                y = bool(interpreter.ev(a, (x0s[0],)))
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                continue

            x = composeBits(a, vocab)
            x = fixLen(x, size)
            x = list(map(float, x))

            if y:
                s = pos
            else:
                s = neg
            if len(s) < n / 2:
                x = torch.as_tensor(x)
                y = torch.as_tensor([float(y)])
                s.append((x, y))
        print(len(seen))
        s = pos + neg
        random.shuffle(s)
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


batch_size = 20

train_ds = Dataset1(800)
test_ds = Dataset1(200)

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

hidden_size = 100
epochs = 10000


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


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        y = y[0]
        with torch.no_grad():
            z = model(x)[0]
        if (y and z > 0.5) or (not y and z <= 0.5):
            n += 1
    return n / len(ds)


interval = epochs // 10
for epoch in range(epochs):
    for bi, (x, y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % interval == 0 and not bi:
            print(
                "%f\t%f\t%f"
                % (loss, accuracy(model, train_ds), accuracy(model, test_ds))
            )
