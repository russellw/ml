# results on a 3 GHz CPU
# counting only FP multiplies/sec
# startup time 1.6 s

# vector length 10
# 3e6

# vector length 100
# 30e6

# vector length 1000
# 294e6

# vector length 10000
# 1428e6

# vector length 100000
# 5780e6

# vector length 1000000
# 3787e6

# vector length 10000000
# 3247e6

import torch
import sys

size = int(sys.argv[1])
times = int(sys.argv[2])
a = torch.randn((size))
b = torch.randn((size))
x = 0.0
for i in range(times):
    c = a @ b
    x += c.item()
print(x)
