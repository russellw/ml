import numpy as np


def esc(c, iters=1000):
    z = complex(0, 0)
    for i in range(iters):
        if abs(z) > 2.0:
            return i
        z = z ** 2 + c
    return iters


def calc(x0=-2.25, y0=-1.5, size=3.0, res=64.0, iters=1000):
    w = []
    for y in np.arange(y0, y0 + size, size / res):
        v = []
        for x in np.arange(x0, x0 + size, size / res):
            c = complex(x, y)
            if esc(c, iters) == iters:
                v.append(1)
            else:
                v.append(0)
        w.append(v)
    return w


if __name__ == "__main__":
    w = calc()
    for v in w:
        for a in v:
            if a:
                print("o", end=" ")
            else:
                print(" ", end=" ")
        print("|")
