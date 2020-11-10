import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# Conway's Game of Life
size = 28


def randboard():
    b = []
    for i in range(size):
        b.append([random.randrange(2) for j in range(size)])
    return b


def get(b, i, j):
    size = len(b)
    if i < 0 or i >= size:
        return 0
    if j < 0 or j >= size:
        return 0
    return b[i][j]


def step(b):
    size = len(b)
    b1 = []
    for i in range(size):
        row = []
        for j in range(size):
            n = 0
            for i1 in range(i - 1, i + 2):
                for j1 in range(j - 1, j + 2):
                    if i1 == i and j1 == j:
                        continue
                    n += get(b, i1, j1)
            if b[i][j]:
                c = n == 2 or n == 3
            else:
                c = n == 3
            row.append(int(c))
        b1.append(row)
    return b1


def printboard(b):
    for row in b:
        for c in row:
            print("@" if c else ".", end=" ")
        print()
    print()


# data


def make_datum():
    b = randboard()
    r = random.randrange(2)
    if r:
        for i in range(3):
            b = step(b)
    return torch.Tensor([b]), r


train_data = [make_datum() for i in range(400)]
test_data = [make_datum() for i in range(100)]

train_loader = torch.utils.data.DataLoader(train_data)
test_loader = torch.utils.data.DataLoader(test_data)

# convolutional neural net


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # https://stackoverflow.com/questions/62628734
        x = self.conv1(x)  # [1, 28, 28] -> [32, 26, 26]
        x = F.relu(x)
        x = self.conv2(x)  # [32, 26, 26] -> [64, 24, 24]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # [64, 24, 24] -> [64, 12, 12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # [64, 12, 12] -> [9216]
        x = self.fc1(x)  # [9216] -> [128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # [128] -> [10]
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


device = torch.device("cpu")
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(3):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
