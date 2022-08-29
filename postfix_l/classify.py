import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from etc import *
import postfix

printTime()
random.seed(0)

xlen = 20

# xs1 = range(10)
xs1 = []
for i in range(10):
    xs1.append(tuple(random.randrange(2) for j in range(10)))


class Dataset1(Dataset):
    def __init__(self, n):
        neg = []
        pos = []
        while len(neg) < n / 2 or len(pos) < n / 2:
            a = postfix.rand(2, xlen // 2)
            if not postfix.good(a, xs1):
                continue
            y = bool(postfix.run(a, xs1[0]))

            a = postfix.compose(a)
            a = fixLen(a, xlen, ")")
            x = []
            for b in a:
                i = postfix.outputVocab.index(b)
                for j in range(len(postfix.outputVocab)):
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
trainDs = Dataset1(ds * 8 // 10)
testDs = Dataset1(ds * 2 // 10)

batchSize = 20

trainDl = DataLoader(trainDs, batch_size=batchSize)
testDl = DataLoader(testDs, batch_size=batchSize)

for x, y in trainDl:
    print(x.shape)
    print(y.shape)
    break

inputSize = xlen * len(postfix.outputVocab)
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
print(sum(p.numel() for p in model.parameters()))


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        y = y[0]
        with torch.no_grad():
            z = model(x)[0]
        if (y and z > 0.5) or (not y and z <= 0.5):
            n += 1
    return n / len(ds)


printTime()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 1000
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
printTime()
