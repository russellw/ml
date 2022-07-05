# https://stackoverflow.com/questions/72867109/what-is-pytorch-dataset-supposed-to-return
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Dataset1(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 80

    def __getitem__(self, i):
        # actual data is blank, just to test the mechanics of Dataset
        return torch.as_tensor([0.0, 0.0, 0.0]), torch.as_tensor([1.0])


train_dataloader = DataLoader(Dataset1(), batch_size=8)

for X, y in train_dataloader:
    print(f"X: {X}")
    print(f"y: {y.shape} {y.dtype} {y}")
    break


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
