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

ways = 5


def oneHot(n, i, s):
    for j in range(n):
        s.append(float(i == j))


class Dataset1(Dataset):
    def __init__(self, n):
        s = []
        d = {}
        for i in range(ways):
            d[i] = 0
        while len(s) < n:
            a = postfix.rand(2, xlen // 2)
            if not postfix.good(a, xs1):
                continue

            try:
                r = int(postfix.run(a, xs1[0]))
            except TypeError:
                continue
            if not 0 <= r < ways:
                continue
            y = []
            oneHot(ways, r, y)
            d[r] += 1

            a = postfix.compose(a)
            a = fixLen(a, xlen, ")")
            x = []
            for b in a:
                oneHot(len(postfix.outputVocab), postfix.outputVocab.index(b), x)

            x = torch.as_tensor(x)
            y = torch.as_tensor(y)
            s.append((x, y))
        random.shuffle(s)
        self.s = s
        print(d)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


ds = 1000
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
            nn.Linear(hiddenSize, ways),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
print(sum(p.numel() for p in model.parameters()))


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        with torch.no_grad():
            z = model(x)
        if torch.argmax(y) == torch.argmax(z):
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
