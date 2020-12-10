import math
import random

import torch
import torch.nn as nn

random.seed(0)

ops = "+", "-", "*", "/", "sqrt"
leaves = 0.0, 1.0
punct = "(", ")"

tokens = {}
for a in ops + leaves + punct:
    tokens[a] = len(tokens)


def arity(o):
    if o == "sqrt":
        return 1
    return 2


def randcode(depth):
    if depth:
        o = random.choice(ops)
        return [o] + [randcode(depth - 1) for i in range(arity(o))]
    return random.choice(leaves)


def evaluate(a):
    if isinstance(a, list) or isinstance(a, tuple):
        try:
            a = list(map(evaluate, a))
            o = a[0]
            x = a[1]
            if o == "sqrt":
                return math.sqrt(x)
            y = a[2]
            return eval(f"x {o} y")
        except (ValueError, ZeroDivisionError):
            return 0.0
    return a


# generate samples
exprs = [randcode(3) for i in range(1000)]
outputs = list(map(evaluate, exprs))

# serialize exprs
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


exprs = list(map(serial, exprs))

# translate tokens to corresponding numbers
def translate(s):
    return [tokens[c] for c in s]


exprs = list(map(translate, exprs))

# pad each string with EOF to make them all the same length
maxlen = max(map(len, exprs))
# maxlen = 50


def pad(s):
    s = s[:maxlen]
    return s + [len(tokens)] * (maxlen - len(s))


exprs = list(map(pad, exprs))

# convert string of numbers to one-hot channels
nchannels = len(tokens) + 1


def one_hot(s):
    r = []
    for i in range(nchannels):
        r.extend([int(x == i) for x in s])
        # r.append([int(x == i) for x in s])
    return r


exprs = list(map(one_hot, exprs))

# convert to x/y tensors
def tensors(exprs, outputs):
    x = torch.tensor(exprs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)
    y = y.view((y.shape[0], 1))
    return x, y


n = len(exprs) * 4 // 5
train_data = tensors(exprs[:n], outputs[:n])
test_data = tensors(exprs[n:], outputs[n:])

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
criterion = nn.MSELoss()
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
