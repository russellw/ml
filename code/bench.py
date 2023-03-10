import time

import torch

n = 1000000000
a = torch.rand(n)
b = torch.rand(n)
r = 0
start = time.time()
for i in range(10000000000 // n):
    a.dot(b)
    r += n * 2
t = time.time() - start
print(r)
print(t)
print(r / t / 1e9)
