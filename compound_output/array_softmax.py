import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


count = 10
bits = 5
size = 1 << bits


class Dataset1(Dataset):
    def __init__(self):
        s = []
        for _ in range(1000):
            x = []
            y = []
            for _ in range(count):
                a = random.randrange(size)

                for c in format(a, "b").zfill(bits):
                    x.append(float(c == "1"))

                y1 = []
                for i in range(size):
                    y1.append(float(a == i))
                y.append(y1)
            x = torch.as_tensor(x)
            y = torch.as_tensor(y)
            s.append((x, y))
        self.s = s

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


trainDs = Dataset1()
testDs = Dataset1()

batchSize = 20
trainDl = DataLoader(trainDs, batch_size=batchSize)
testDl = DataLoader(testDs, batch_size=batchSize)
for x, y in trainDl:
    print(x.shape)
    print(y.shape)
    break


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        batchSize = x.data.size(0)
        shape = (batchSize,) + self.shape
        return x.view(*shape)


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
            nn.Linear(hiddenSize, count * size),
            View(count, size),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
print(sum(p.numel() for p in model.parameters()))


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        # make input sample shape match a mini batch
        # for the sake of things like softmax that cause the model
        # to expect a specific shape
        x = x.unsqueeze(0)

        # this is just for reporting, not part of training
        # so we don't need to track gradients here
        with torch.no_grad():
            z = model(x)

            # conversely, the model will return a batch-shaped output
            # so unwrap it for comparison with the unwrapped expected output
            z = z[0]

        # at this point, if the output were a scalar mapped to one-hot
        # we could use a simple argmax comparison
        # but it is an array thereof
        # which makes comparison a little more complex
        assert y.shape[0] == count
        assert z.shape[0] == count
        for i in range(0, count):
            if torch.argmax(y[i]) == torch.argmax(z[i]):
                n += 1
    return n / (len(ds) * count)


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
            print(
                f"{epoch}\t{loss}\t{accuracy(model, trainDs)}\t{accuracy(model, testDs)}"
            )
