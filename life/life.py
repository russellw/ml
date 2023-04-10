import argparse
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

    def new_cell(self, x, y):
        n = 0
        for y2 in range(y - 1, y + 2):
            for x2 in range(x - 1, x + 2):
                if x2 != x or y2 != y:
                    n += self[x2, y2]
        return n == 3 or n == 2 and self[x, y]

    def __repr__(self):
        x0, y0, x1, y1 = self.bound()
        return f"{x0,y0} -> {x1,y1}"

    def run(self, steps=1):
        for step in range(steps):
            x0, y0, x1, y1 = self.bound()
            x0 -= 1
            y0 -= 1
            x1 += 1
            y1 += 1
            new = set()
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if self.new_cell(x, y):
                        new.add((x, y))
            self.data = new


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--density", help="density of random grid", type=float, default=0.5
    )
    parser.add_argument("-g", "--steps", help="number of steps", type=int, default=1000)
    parser.add_argument("-r", "--rand", help="random pattern size", type=int)
    parser.add_argument("-s", "--seed", help="random number seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    g = None
    if args.rand is not None:
        g = randgrid(args.rand, args.density)
    if g:
        g.run(args.steps)
        prn(g)
