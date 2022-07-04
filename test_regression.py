import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
X_train = np.array([[0.0, 0.0], [0.0, 0.3], [0.3, 0.0], [0.3, 0.3]], dtype=np.float32)
y_train = np.array([[0.0], [0.3], [0.3], [0.6]], dtype=np.float32)

# torch tensors
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

# hyperparameters
in_features = X_train.shape[1]
hidden_size = 100
epochs = 1000

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.layers(x)


model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train
# sometimes this converges on zero error, sometimes not
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
        print(f"{epoch:6d} {cost.item():10f}")
