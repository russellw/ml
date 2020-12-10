import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        return x


model = Net()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MSELoss()

# test
x = torch.tensor(0.0)
y = torch.tensor(0.5)
output = model(x)
cost = criterion(output, y)
print(f"       {cost.item():10f}")
