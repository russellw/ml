import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 4),)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    print(epoch, end="")
    t = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = [i, j, k]
                y = [0] * 4
                y[sum(x)] = 1
                x = torch.as_tensor(x, dtype=torch.float32)
                y = torch.as_tensor(y, dtype=torch.float32)

                loss = criterion(model(x), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("\t%f" % loss, end="")
                t += loss
    print("\t%f" % t)
