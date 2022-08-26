import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from etc import *
import interpreter
import rand

random.seed(0)

flen = 10

# xs1 = range(10)
xs1 = []
for i in range(10):
    xs1.append(tuple(random.randrange(2) for j in range(10)))


class Dataset1(Dataset):
    def __init__(self, n):
        neg = []
        pos = []
        while len(neg) < n / 2 or len(pos) < n / 2:
            p = rand.mk(2, flen)
            try:
                if not interpreter.good(p, xs1):
                    continue
                p = rand.rmDead(p)
                y = bool(interpreter.run(p, xs1[0]))
            except (
                IndexError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                continue

            x = []
            for i in range(rand.fcount):
                k = fname(i)
                if k in p:
                    f = p[k]
                else:
                    f = ()
                f = fixLen(f, flen, "end")
                for a in f:
                    i = rand.vocab.index(a)
                    for j in range(len(rand.vocab)):
                        x.append(float(i == j))
                    continue

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


ds = 10000
trainDs = Dataset1(ds * 10 // 8)
testDs = Dataset1(ds * 10 // 2)

batchSize = 20

trainDl = DataLoader(trainDs, batch_size=batchSize)
testDl = DataLoader(testDs, batch_size=batchSize)

for x, y in trainDl:
    print(x.shape)
    print(y.shape)
    break

inputSize = rand.fcount * flen * len(rand.vocab)
hiddenSize = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
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


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        y = y[0]
        with torch.no_grad():
            z = model(x)[0]
        if (y and z > 0.5) or (not y and z <= 0.5):
            n += 1
    return n / len(ds)


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 100
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
