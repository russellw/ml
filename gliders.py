import random

# Conway's Game of Life
size = 8


def blankboard():
    b = []
    for i in range(size):
        b.append([0] * size)
    return b


def randboard():
    b = []
    for i in range(size):
        b.append([random.randrange(2) for j in range(size)])
    return b


def randpattern(sz):
    b = blankboard()
    for i in range(sz):
        for j in range(sz):
            b[i][j] = random.randrange(2)
    return b


def neighborhood8(i, j):
    for i1 in range(i - 1, i + 2):
        for j1 in range(j - 1, j + 2):
            if i1 == i and j1 == j:
                continue
            yield (i1, j1)


def step(b):
    b1 = []
    for i in range(size):
        row = []
        for j in range(size):
            n = 0
            for i1, j1 in neighborhood8(i, j):
                n += b[i1 % size][j1 % size]
            if b[i][j]:
                c = n == 2 or n == 3
            else:
                c = n == 3
            row.append(int(c))
        b1.append(row)
    return b1


def steps(b, n):
    for i in range(n):
        b = step(b)
    return b


def printboard(b):
    for row in b:
        for c in row:
            print("@" if c else ".", end=" ")
        print()
    print()


def popcount(b):
    n = 0
    for row in b:
        n += sum(row)
    return n


def shift(b, di, dj):
    b1 = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(b[(i - di) % size][(j - dj) % size])
        b1.append(row)
    return b1


def is_still(b):
    n = popcount(b)
    if n == 0:
        return 0
    b1 = step(b)
    return b1 == b


def is_blinker(b):
    n = popcount(b)
    if n == 0:
        return 0
    b1 = step(b)
    if b1 == b:
        return 0
    b1 = step(b1)
    return b1 == b


def is_glider(b):
    n = popcount(b)
    if n == 0:
        return 0
    b1 = b
    for i in range(4):
        b1 = step(b1)
        if popcount(b1) != n:
            continue
        for i1, j1 in neighborhood8(0, 0):
            if shift(b, i1, j1) == b1:
                return 1


# search
for i in range(1000000):
    b = randpattern(5)
    b = steps(b, 10)
    if is_glider(b):
        printboard(b)
        print(i)
        break
