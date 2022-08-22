import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from etc import *
import interpreter
import rand

maxLen = 10
maxBits = maxLen * bitLen(len(rand.vocab))
xs1 = range(10)


class Dataset1(Dataset):
    def __init__(self, n):
        neg = []
        pos = []
        while len(neg) < n / 2 or len(pos) < n / 2:
            f = rand.mk(2, maxLen)

            try:
                if not interpreter.good(f, xs1):
                    continue
                y = bool(interpreter.run(f, (xs1[0],)))
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                continue

            x = composeBits(f, rand.vocab)
            x = fixLen(x, maxBits)
            x = list(map(float, x))

            if y:
                s = pos
            else:
                s = neg
            if len(s) < n / 2:
                x = torch.as_tensor(x)
                y = torch.as_tensor([float(y)])
                s.append((x, y))
        s = neg + pos
        random.shuffle(s)
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


ds = 1000
trainDs = Dataset1(ds * 10 // 8)
testDs = Dataset1(ds * 10 // 2)

batchSize = 20

trainDl = DataLoader(trainDs, batch_size=batchSize)
testDl = DataLoader(testDs, batch_size=batchSize)

for x, y in trainDl:
    print(x.shape)
    print(y.shape)
    break

hiddenSize = 100
epochs = 10000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(maxBits, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, 1),
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
for epoch in range(epochs + 1):
    for bi, (x, y) in enumerate(trainDl):
        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % interval == 0 and not bi:
            print(
                f"{epoch}\t{loss}\t{accuracy(model, trainDs)}\t{accuracy(model, testDs)}"
            )
