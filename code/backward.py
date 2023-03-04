import torch

x = torch.tensor([[1.0, -1.0], [1.0, 1.0]], requires_grad=True)
out = x.pow(2).sum()
out.backward()
print(x.grad)
print(x)
