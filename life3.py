import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

random.seed(0)

# Conway's Game of Life
def neighborhood8(i, j):
    for i1 in range(i - 1, i + 2):
        for j1 in range(j - 1, j + 2):
            if i1 == i and j1 == j:
                continue
            yield (i1, j1)


def exists_col(rows, j):
    for i in range(len(rows)):
        if j < len(rows[i]) and rows[i][j]:
            return 1


def wid(rows):
    if not rows:
        return 0
    return max([len(row) for row in rows])


def trim_row(row):
    i = len(row)
    while i and not row[i - 1]:
        i -= 1
    return row[:i]


def trim_rows(rows):
    # trim cols
    r = [trim_row(row) for row in rows]

    # count blank south
    i1 = len(r)
    while i1 and r[i1 - 1]:
        i1 -= 1

    # count blank north
    i0 = 0
    while i0 < i1 and r[i0]:
        i0 += 1

    # width
    j1 = wid(r)

    # count blank west
    j0 = 0
    while j0 < j1 and not exists_col(r, j0):
        j0 += 1

    # trim rows
    r = r[i0:i1]

    # trim cols
    r = [row[j0:] for row in r]

    # return with notice of new origin
    return r, i0, j0


def square(rows, size):
    rows = rows[:size]
    rows.extend([[]] * (size - len(rows)))
    for i in range(len(rows)):
        row = rows[i]
        row = row[:size]
        row.extend([0] * (size - len(row)))
        rows[i] = row
    return rows


class Grid:
    def __init__(self, data="", origin=(0, 0)):
        rows = data
        if isinstance(data, str):
            data = data.strip()
            rows = []
            for s in data.split("\n"):
                rows.append([c != "." for c in s])
            j1 = wid(rows)
            for row in rows:
                row.extend([False] * (j1 - len(row)))
        rows, i0, j0 = trim_rows(rows)
        self.rows = rows
        self.origin = origin[0] + i0, origin[1] + j0

    def __bool__(self):
        return self.popcount() > 0

    def __eq__(self, other):
        return self.origin == other.origin and self.rows == other.rows

    def __getitem__(self, key):
        i = key[0]
        if not (0 <= i < len(self.rows)):
            return False
        row = self.rows[i]
        j = key[1]
        if not (0 <= j < len(row)):
            return False
        return row[j]

    def __repr__(self):
        r = []
        if self.origin != (0, 0):
            r.append("(%d, %d)\n" % self.origin)
        for row in self.rows:
            for c in row:
                r.append("O" if c else ".")
                r.append(" ")
            r.append("\n")
        return "".join(r)

    def __xor__(self, other):
        i0 = min(self.origin[0], other.origin[0])
        i1 = max(self.origin[0] + len(self.rows), other.origin[0] + len(other.rows))
        j0 = min(self.origin[1], other.origin[1])
        j1 = max(self.origin[1] + wid(self.rows), other.origin[1] + wid(other.rows))
        r = []
        for i in range(i0, i1):
            row = []
            for j in range(j0, j1):
                row.append(self[i, j] ^ other[i, j])
            r.append(row)
        return Grid(r, (i0, j0))

    def home(self):
        return Grid(self.rows)

    def popcount(self):
        n = 0
        for row in self.rows:
            n += sum(row)
        return n

    def step(self):
        i1 = len(self.rows)
        j1 = wid(self.rows)
        rows = []
        for i in range(-1, i1 + 1):
            row = []
            for j in range(-1, j1 + 1):
                n = 0
                for i1, j1 in neighborhood8(i, j):
                    n += self[i1, j1]
                if self[i, j]:
                    c = n == 2 or n == 3
                else:
                    c = n == 3
                row.append(c)
            rows.append(row)
        return Grid(rows, (self.origin[0] - 1, self.origin[1] - 1))

    def steps(self, n):
        g = self
        for i in range(n):
            g = g.step()
        return g

    def tensor(self, size):
        return torch.Tensor([square(self.rows, size)])


def randgrid(size):
    rows = []
    for i in range(size):
        rows.append([random.randrange(2) for j in range(size)])
    return Grid(rows)


# data


def make_datum():
    g = randgrid(28)
    r = random.randrange(2)
    if r:
        g = g.step()
    return g.tensor(28), r


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
