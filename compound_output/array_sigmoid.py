import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


count = 10
bits = 5


class Dataset1(Dataset):
    def __init__(self):
        s = []
        for _ in range(1000):
            x = []
            y = []
            for _ in range(count):
                a = random.randrange(1 << bits)

                for c in format(a, "b").zfill(bits):
                    x.append(float(c == "1"))

                for i in range(1 << bits):
                    y.append(float(a == i))
            x = torch.as_tensor(x)
            y = torch.as_tensor(y)
            s.append((x, y))
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


trainDs = Dataset1()

batchSize = 20
trainDl = DataLoader(trainDs, batch_size=batchSize)

for x, y in trainDl:
    print(x.shape)
    print(y.shape)
    break

hiddenSize = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(count * bits, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, count * (1 << bits)),
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
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        with torch.no_grad():
            z = model(x)
        if torch.argmax(y) == torch.argmax(z):
            n += 1
    return n / len(ds)


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 10000
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
            print(f"{epoch}\t{loss}\t{accuracy(model, trainDs)}")
