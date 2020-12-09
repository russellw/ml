import os
import random

import torch
import torch.nn as nn

# data
java_sources = []
python_sources = []


def do_file(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".java":
        java_sources.append(open(filename).read())
    elif ext == ".py":
        python_sources.append(open(filename).read())


for root, dirs, files in os.walk("."):
    for fname in files:
        do_file(os.path.join(root, fname))

# random order to avoid bias
random.shuffle(java_sources)
random.shuffle(python_sources)

# same number of cases for each class
n = min(len(java_sources), len(python_sources))
java_sources = java_sources[:n]
python_sources = python_sources[:n]

# for efficiency, only create channels for characters that are actually used
chars_used = set()
for s in java_sources + python_sources:
    for c in s:
        chars_used.add(c)
chars_used = sorted(list(chars_used))
chars = {}
for i in range(len(chars_used)):
    chars[chars_used[i]] = i

# translate characters to corresponding numbers
def translate(s):
    return [chars[c] for c in s]


java_sources = list(map(translate, java_sources))
python_sources = list(map(translate, python_sources))

# pad each string with EOF to make them all the same length
maxlen = max(map(len, java_sources + python_sources))
maxlen = 50


def pad(s):
    s = s[:maxlen]
    return s + [len(chars)] * (maxlen - len(s))


java_sources = list(map(pad, java_sources))
python_sources = list(map(pad, python_sources))

# convert string of numbers to one-hot channels
nchannels = len(chars) + 1


def one_hot(s):
    r = []
    for i in range(nchannels):
        r.extend([int(x == i) for x in s])
        # r.append([int(x == i) for x in s])
    return r


java_sources = list(map(one_hot, java_sources))
python_sources = list(map(one_hot, python_sources))

# convert data to x/y tensors
def interleave(s, t):
    assert len(s) == len(t)
    r = []
    for i in range(len(s)):
        r.append(s[i])
        r.append(t[i])
    return r


def tensors(no, yes):
    assert len(no) == len(yes)
    x = torch.tensor(interleave(no, yes), dtype=torch.float32)
    y = torch.tensor([0, 1] * len(no), dtype=torch.float32)
    y = y.view((y.shape[0], 1))
    return x, y


n = len(java_sources) * 3 // 4
train_data = tensors(java_sources[:n], python_sources[:n])
test_data = tensors(java_sources[n:], python_sources[n:])

# hyperparameters
in_features = maxlen * nchannels
hidden_size = 100

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(in_features, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.out(x)


model = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train
x, y = train_data
epochs = 10
for epoch in range(1, epochs + 1):
    # forward
    output = model(x)
    cost = criterion(output, y)

    # backward
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print progress
    if epoch % (epochs // 10) == 0:
        print(f"{epoch:6d} {cost.item():10f}")
print()

# test
x, y = test_data
output = model(x)
cost = criterion(output, y)
print(f"       {cost.item():10f}")
