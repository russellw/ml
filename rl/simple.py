import torch
from torch import nn

size = 11
hidden_size = 20
actions = 2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions),
        )

    def forward(self, x):
        return self.layers(x)
