# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
import torch
from torch import nn

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

output = loss(input, target)
print(input)
print(target)
print(output)
print()

for i in range(1000):
    output.backward()
    output = loss(input, target)
print(input)
print(target)
print(output)
print()

# Example of target with class probabilities
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)

output = loss(input, target)
print(input)
print(target)
print(output)
print()

for i in range(1000):
    output.backward()
    output = loss(input, target)
print(input)
print(target)
print(output)
print()
