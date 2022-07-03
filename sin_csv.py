import math

import numpy as np

print("x,y")
for x in np.arange(-10, 10, 0.1):
    y = math.sin(x)
    print(f"{x},{y}")
