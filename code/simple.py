import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, 1),)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)

print(model(torch.as_tensor([1, 1, 1], dtype=torch.float32)))
print(model(torch.as_tensor([[1, 1, 1]], dtype=torch.float32)))
