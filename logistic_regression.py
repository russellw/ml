import argparse
import csv

import numpy as np
import torch
import torch.nn as nn


# args
parser = argparse.ArgumentParser(description="simple logistic regression")
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()

# data
X_train = []
y_train = []
with open(args.filename, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for r in reader:
        X_train.append(r[1:-1])
        y_train.append(r[-1])

# inputs
in_features = len(X_train[0])
print(f"X: {len(X_train)} x {in_features}")

X_train = np.array(X_train, dtype=np.float32)
X_tensor = torch.tensor(X_train)

# output
ys = set()
for y in y_train:
    ys.add(y)
ys = sorted(list(ys))
out_features = len(ys)
print(f"y: {len(y_train)}; {out_features} possible values")

y_frequency = {}
for y in ys:
    y_frequency[y] = 0
for y in y_train:
    y_frequency[y] += 1
print(y_frequency)
print()

yd = {}
for y in y_train:
    if y not in yd:
        yd[y] = len(yd)

for i in range(len(y_train)):
    y_train[i] = yd[y_train[i]]

y_train = np.array(y_train, dtype=np.uint8)
y_tensor = torch.tensor(y_train, dtype=torch.long)

# model
model = nn.Linear(in_features, out_features)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train
epochs = 5000
print("training")
for epoch in range(1, epochs + 1):
    # forward
    output = model(X_tensor)
    cost = criterion(output, y_tensor)

    # backward
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print progress
    if epoch % (epochs // 50) == 0:
        print(f"{epoch:6d}  {cost.item():6f}")
print()

# accuracy on training data
with torch.no_grad():
    output = model(X_tensor)
    print("model")
    print(output)
    print()

    predicted = torch.argmax(output.data, 1)
    print("predicted")
    print(predicted)
    print(predicted == y_tensor)
    print()

    print("accuracy if used as classifier")
    correct = (predicted == y_tensor).sum()
    total = y_tensor.size(0)
    print(
        f"logistic classifier: {correct:6d} / {total:6d} = {correct * 100 / total:.3f}%"
    )

    correct = 0
    for y in ys:
        correct = max(correct, y_frequency[y])
    print(
        f"guess commonest    : {correct:6d} / {total:6d} = {correct * 100 / total:.3f}%"
    )

    correct = total // len(ys)
    print(
        f"guess random       : {correct:6d} / {total:6d} = {correct * 100 / total:.3f}%"
    )
