import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


bits = 5


class Dataset1(Dataset):
    def __init__(self):
        s = []
        for i in range(1 << bits):
            x = []
            for c in format(i, "b").zfill(bits):
                x.append(float(c == "1"))

            y = [float(i % 3 != 0)]

            x = torch.as_tensor(x)
            y = torch.as_tensor(y)
            s.append((x, y))
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


trainDs = Dataset1()

batchSize = (1 << bits) // 5
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
            nn.Linear(bits, hiddenSize),
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
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        with torch.no_grad():
            z = model(x)
        print()
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(x)
        print(y)
        print(z)
        if (y and z > 0.5) or (not y and z <= 0.5):
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
