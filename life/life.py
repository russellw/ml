import argparse
import os
import random


def bound(a):
    if not a:
        return 0, 0, 0, 0
    x0, y0 = next(iter(a))
    x1, y1 = x0, y0
    for x, y in a:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x, x1)
        y1 = max(y, y1)
    return x0, y0, x1 + 1, y1 + 1


def matrix(a, *q):
    if not q:
        q = bound(a)
    x0, y0, x1, y1 = q

    w = []
    for y in range(y0, y1):
        v = []
        for x in range(x0, x1):
            v.append((x, y) in a)
        w.append(v)
    return w


def run(a, steps=1):
    for step in range(steps):
        x0, y0, x1, y1 = bound(a)
        x0 -= 1
        y0 -= 1
        x1 += 1
        y1 += 1

        b = set()
        for y in range(y0, y1):
            y_minus_1 = y - 1
            y_plus_1 = y + 1
            for x in range(x0, x1):
                n = (
                    ((x - 1, y_minus_1) in a)
                    + ((x - 1, y) in a)
                    + ((x - 1, y_plus_1) in a)
                    + ((x, y_minus_1) in a)
                    + ((x, y_plus_1) in a)
                    + ((x + 1, y_minus_1) in a)
                    + ((x + 1, y) in a)
                    + ((x + 1, y_plus_1) in a)
                )
                if n == 3 or n == 2 and (x, y) in a:
                    b.add((x, y))
        a = b
    return a


def read_rle(file):
    s = open(file).read()
    i = 0
    x = 0
    y = 0
    a = set()
    while i < len(s):
        # comment
        if s[i] in "#x":
            while s[i] != "\n":
                i += 1

        # space
        if s[i].isspace():
            i += 1
            continue

        # end
        if s[i] == "!":
            break

        # run count
        n = 1
        if s[i].isdigit():
            j = i
            while s[i].isdigit():
                i += 1
            n = int(s[j:i])

        # tag
        t = s[i]
        i += 1
        if t == "b":
            x += n
        elif t == "o":
            for j in range(n):
                a.add((x, y))
                x += 1
        elif t == "$":
            x = 0
            y += n
        else:
            raise Exception(t)
    return a


def read(file):
    ext = os.path.splitext(file)[1]
    if ext == ".cells":
        return read_plaintext(file)
    if ext == ".rle":
        return read_rle(file)
    raise Exception(file)


def read_plaintext(file):
    y = 0
    a = set()
    for s in open(file).readlines():
        # comment
        if s.startswith("!"):
            continue

        # cells
        x = 0
        for c in s:
            if c.isspace():
                continue

            if c == ".":
                pass
            elif c == "O":
                a.add((x, y))
            else:
                raise Exception(s)
            x += 1
        y += 1
    return a


def rand(size, density=0.5):
    a = set()
    for y in range(size):
        for x in range(size):
            if random.random() < density:
                a.add((x, y))
    return a


def prn(a):
    x0, y0, x1, y1 = bound(a)
    print(f"{x0},{y0} -> {x1},{y1}")
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x, y) in a:
                print("O", end=" ")
            else:
                print(".", end=" ")
        print()
    print()


if __name__ == "__main__":
    a = set()
    a.add((0, 0))
    assert bound(a) == (0, 0, 1, 1)

    a = run(a)
    assert len(a) == 0
    assert bound(a) == (0, 0, 0, 0)

    a.add((0, 0))
    a.add((0, 1))
    a.add((1, 0))
    assert bound(a) == (0, 0, 2, 2)

    a = run(a)
    assert len(a) == 4
    assert bound(a) == (0, 0, 2, 2)
    assert matrix(a) == [[1, 1], [1, 1]]
    assert matrix(a, 0, 0, 3, 3) == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--density", help="density of random grid", type=float, default=0.5
    )
    parser.add_argument("-g", "--steps", help="number of steps", type=int, default=100)
    parser.add_argument("-r", "--rand", help="random pattern size", type=int)
    parser.add_argument("-s", "--seed", help="random number seed", type=int)
    parser.add_argument("file", nargs="?")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    a = None
    if args.rand is not None:
        a = rand(args.rand, args.density)
    if args.file:
        a = read(args.file)
    if a:
        a = run(a, args.steps)
        prn(a)
