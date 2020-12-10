import random

import torch
import torch.nn as nn

random.seed(0)

ops = "and", "or", "not"
leaves = True, False
punct = "(", ")"

tokens = {}
for a in ops + leaves + punct:
    tokens[a] = len(tokens)


def arity(o):
    if o == "not":
        return 1
    return 2


def randcode(depth):
    if depth:
        o = random.choice(ops)
        return [o] + [randcode(depth - 1) for i in range(arity(o))]
    return random.choice(leaves)


def evaluate(a):
    if isinstance(a, list) or isinstance(a, tuple):
        a = list(map(evaluate, a))
        o = a[0]
        x = a[1]
        if arity(o) == 1:
            return eval(f"{o} x")
        y = a[2]
        return eval(f"x {o} y")
    return a


# gather samples
n = 1000
no = []
yes = []
while len(no) < n or len(yes) < n:
    a = randcode(3)
    r = yes if evaluate(a) else no
    if len(r) < n:
        r.append(a)

# serialize
def serial(a):
    r = []

    def rec(a):
        if isinstance(a, list) or isinstance(a, tuple):
            r.append("(")
            for b in a:
                rec(b)
            r.append(")")
            return
        r.append(a)

    rec(a)
    return r


no = list(map(serial, no))
yes = list(map(serial, yes))

# translate tokens to corresponding numbers
def translate(s):
    return [tokens[c] for c in s]


no = list(map(translate, no))
yes = list(map(translate, yes))

# pad each string with EOF to make them all the same length
maxlen = max(map(len, no + yes))
# maxlen = 50


def pad(s):
    s = s[:maxlen]
    return s + [len(tokens)] * (maxlen - len(s))


no = list(map(pad, no))
yes = list(map(pad, yes))

# convert string of numbers to one-hot channels
nchannels = len(tokens) + 1


def one_hot(s):
    r = []
    for i in range(nchannels):
        r.extend([int(x == i) for x in s])
        # r.append([int(x == i) for x in s])
    return r


no = list(map(one_hot, no))
yes = list(map(one_hot, yes))

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


n = len(no) * 3 // 4
train_data = tensors(no[:n], yes[:n])
test_data = tensors(no[n:], yes[n:])

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
epochs = 1000
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
