import math
import random
import time

import psutil
import skopt
import torch
import torch.nn as nn

process = psutil.Process()
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


exprs = [randcode(3) for i in range(1000)]
outputs = list(map(evaluate, exprs))


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


def translate(s):
    return [tokens[c] for c in s]


exprs = list(map(translate, exprs))

# pad each string with EOF to make them all the same length
maxlen = max(map(len, exprs))
print(f"maxlen: {maxlen}")
print()


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


def tensors(exprs, outputs):
    x = torch.tensor(exprs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)
    y = y.view((y.shape[0], 1))
    return x, y


n = len(exprs)
train_x, train_y = tensors(exprs[: n * 3 // 5], outputs[: n * 3 // 5])
valid_x, valid_y = tensors(
    exprs[n * 3 // 5 : n * 4 // 5], outputs[n * 3 // 5 : n * 4 // 5]
)
test_x, test_y = tensors(exprs[n * 4 // 5 :], outputs[n * 4 // 5 :])

optim_names = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "ASGD",
    "RMSprop",
    "Rprop",
    "SGD",
]

# LBFGS needs an extra closure parameter
# SparseAdam does not support dense gradients, please consider Adam instead
optims = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}

space = [
    skopt.space.Integer(1, 1000, name="hidden1"),
    skopt.space.Categorical(optim_names, name="optim"),
    skopt.space.Real(10 ** -4, 0.5, "log-uniform", name="lr"),
]


def hparam(hparams, name):
    for i in range(len(hparams)):
        if space[i].name == name:
            return hparams[i]
    raise ValueError(name)


def optim(hparams):
    return optims[hparam(hparams, "optim")]


class Net(nn.Module):
    def __init__(self, hidden1):
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(nchannels * maxlen, hidden1)
        self.out = nn.Linear(hidden1, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.out(x)


criterion = nn.MSELoss()
count = 0


def train(hparams):
    global count
    print(count)
    if isinstance(count, int):
        count += 1
    print(hparams)

    model = Net(hparam(hparams, "hidden1"))
    optimizer = optim(hparams)(model.parameters(), lr=hparam(hparams, "lr"))
    print(f"{process.memory_info().rss:,} bytes")

    epochs = 1000
    for epoch in range(epochs + 1):
        # print progress
        if epoch % (epochs // 10) == 0:
            train_cost = criterion(model(train_x), train_y).item()
            valid_cost = criterion(model(valid_x), valid_y).item()
            test_cost = criterion(model(test_x), test_y).item()
            print(f"{epoch:6d} {train_cost:10f} {valid_cost:10f} {test_cost:10f}")

        # forward
        output = model(train_x)
        cost = criterion(output, train_y)

        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    print()
    return valid_cost


start = time.time()

# search hyperparameters
# n_calls defaults to 100
res = skopt.gp_minimize(train, space, n_calls=100)

# train once more with best hyperparameters
count = "final"
train(res.x)

print(f"{time.time() - start:.3f} seconds")
