import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid(),)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = torch.as_tensor([i, j, k], dtype=torch.float32)
                y = torch.as_tensor([int(i + j + k > 1.5)], dtype=torch.float32)
                print(x)
                print(y)
                print(model(x))

                loss = criterion(model(x), y)
                print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for p in model.parameters():
                    print(p)

                print()
