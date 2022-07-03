import math

import matplotlib.pyplot as plt
import numpy as np

a = np.empty([10, 10], dtype=np.float32)
for y in range(10):
    y1 = y - 5
    for x in range(10):
        x1 = x - 5
        z = math.sin(math.sqrt(x1 ** 2 + y1 ** 2))
        a[y, x] = z

plt.contour(a)
plt.show()
