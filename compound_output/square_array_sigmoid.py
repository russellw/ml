import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def oneHot(n, i, s):
    for j in range(n):
        s.append(float(i == j))


size = 12


class Dataset1(Dataset):
    def __init__(self):
        s = []
        for _ in range(1000):
            a = random.randrange(10 ** size)

            x = []
            for c in str(a).zfill(size):
                oneHot(10, int(c), x)

            y = []
            for c in str(a ** 2).zfill(size * 2):
                oneHot(10, int(c), y)

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


hiddenSize = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size * 10, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, size * 2 * 10),
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
        for i in range(0, y.shape[0], 10):
            if torch.argmax(y[i : i + 10]) == torch.argmax(z[i : i + 10]):
                n += 1
    return n / (len(ds) * size * 2)


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
