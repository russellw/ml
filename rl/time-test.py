import time

steps = 10000000
size = 8

v = []
for i in range(8):
    v.append([0] * 8)

start = time.time()
n = 0
i = 4
j = 4
for step in range(steps):
    n += v[i][j]
print(time.time() - start)


def at(i, j):
    return i * size + j


v = [0] * 64

start = time.time()
n = 0
i = 4
j = 4
for step in range(steps):
    n += v[at(i, j)]
print(time.time() - start)
