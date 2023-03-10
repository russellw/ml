import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn

# this does not work
# ModuleNotFoundError: No module named 'win32con'
# (torch2) C:\ml\code>conda install win32con
# Collecting package metadata (current_repodata.json): done
# Solving environment: failed with initial frozen solve. Retrying with flexible solve.
# Collecting package metadata (repodata.json): failed


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, 1),)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)

for p in model.parameters():
    print(p)
print()

print(model(torch.as_tensor([1, 1, 1], dtype=torch.float32)))
print(model(torch.as_tensor([[1, 1, 1]], dtype=torch.float32)))

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

flops = FlopCountAnalysis(model, input)
print(flops.total())
print(flops.by_operator())
print(flops.by_module())
print(flops.by_module_and_operator())
