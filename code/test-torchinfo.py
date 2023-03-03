import torch
from torch import nn
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, 1),)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)

batch_size = 16
summary(model, input_size=(3,))
summary(model, input_size=(batch_size, 3))
