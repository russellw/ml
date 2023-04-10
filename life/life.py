import argparse
import os
import random


class Grid:
    def __init__(self):
        self.data = set()

    def __setitem__(self, xy, c):
        if c:
            self.data.add(xy)
        else:
            self.data.discard(xy)

    def __getitem__(self, xy):
        return xy in self.data

    def bound(self):
        if not self.data:
            return 0, 0, 0, 0
        x0, y0 = next(iter(self.data))
        x1, y1 = x0, y0
        for x, y in self.data:
            x0 = min(x0, x)
            y0 = min(y0, y)
            x1 = max(x, x1)
            y1 = max(y, y1)
        x1 += 1
        y1 += 1
        return x0, y0, x1, y1

    def popcount(self):
        return len(self.data)

    def __repr__(self):
        x0, y0, x1, y1 = self.bound()
        return f"{x0},{y0} -> {x1},{y1}"

    def get_data(self, *b):
        if not b:
            b = self.bound()
        x0, y0, x1, y1 = b
        q = []
        for y in range(y0, y1):
            r = []
            for x in range(x0, x1):
                r.append(self[x, y])
            q.append(r)
        return q

    def run(self, steps=1):
        for step in range(steps):
            x0, y0, x1, y1 = self.bound()
            x0 -= 1
            y0 -= 1
            x1 += 1
            y1 += 1

            d = self.data
            new = set()
            for y in range(y0, y1):
                for x in range(x0, x1):
                    n = (
                        ((x - 1, y - 1) in d)
                        + ((x - 1, y) in d)
                        + ((x - 1, y + 1) in d)
                        + ((x, y - 1) in d)
                        + ((x, y + 1) in d)
                        + ((x + 1, y - 1) in d)
                        + ((x + 1, y) in d)
                        + ((x + 1, y + 1) in d)
                    )
                    if n == 3 or n == 2 and self[x, y]:
                        new.add((x, y))
            self.data = new


def read_rle(file):
    s = open(file).read()
    i = 0
    x = 0
    y = 0
    g = Grid()
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
                g[x, y] = 1
                x += 1
        elif t == "$":
            x = 0
            y += n
        else:
            raise Exception(t)
    return g


def read(file):
    ext = os.path.splitext(file)[1]
    if ext == ".cells":
        return read_plaintext(file)
    if ext == ".rle":
        return read_rle(file)
    raise Exception(file)


def read_plaintext(file):
    y = 0
    g = Grid()
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
                g[x, y] = 1
            else:
                raise Exception(s)
            x += 1
        y += 1
    return g


def randgrid(size, density=0.5):
    g = Grid()
    for y in range(size):
        for x in range(size):
            if random.random() < density:
                g[x, y] = 1
    return g


def prn(g):
    print(g)
    x0, y0, x1, y1 = g.bound()
    for y in range(y0, y1):
        for x in range(x0, x1):
            if g[x, y]:
                print("O", end=" ")
            else:
                print(".", end=" ")
        print()
    print()


if __name__ == "__main__":
    g = Grid()
    assert g.popcount() == 0

    g[0, 0] = 1
    assert g.popcount() == 1
    assert g.bound() == (0, 0, 1, 1)

    g.run()
    assert g.popcount() == 0
    assert g.bound() == (0, 0, 0, 0)

    g[0, 0] = 1
    g[0, 1] = 1
    g[1, 0] = 1
    assert g.popcount() == 3
    assert g.bound() == (0, 0, 2, 2)

    g.run()
    assert g.popcount() == 4
    assert g.bound() == (0, 0, 2, 2)
    assert g.get_data() == [[1, 1], [1, 1]]
    assert g.get_data(0, 0, 3, 3) == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--density", help="density of random grid", type=float, default=0.5
    )
    parser.add_argument("-g", "--steps", help="number of steps", type=int, default=1000)
    parser.add_argument("-r", "--rand", help="random pattern size", type=int)
    parser.add_argument("-s", "--seed", help="random number seed", type=int)
    parser.add_argument("file", nargs="?")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    g = None
    if args.rand is not None:
        g = randgrid(args.rand, args.density)
    if args.file:
        g = read(args.file)
    if g:
        g.run(args.steps)
        prn(g)
