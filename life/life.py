import random


class Grid:
    def __init__(self):
        self.data = set()

    def __setitem__(self, xy, c):
        if c:
            self.data.add(xy)
        else:
            self.data.discard(xy)

    def __getitem__(self, x, y):
        return (x, y) in self.data

    def bound(self):
        if not self.data:
            return 0, 0, 0, 0
        x0, y0 = next(iter(self.data))
        x1, y1 = x0, y0
        for (x, y) in self.data:
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
                if x2 or y2:
                    n += self[x2, y2]
        return n == 3 or n == 2 and self[x, y]

    def run(self, steps=1):
        for step in range(steps):
            x0, y0, x1, y1 = self.bound()
            new = set()
            for y in range(y0 - 1, y1 + 1):
                for x in range(x0 - 1, x1 + 1):
                    if self.new_cell(x, y):
                        new.add((x, y))
            self.data = new


def randgrid(size):
    rows = []
    for i in range(size):
        rows.append([random.randrange(2) for j in range(size)])
    return Grid(rows)


if __name__ == "__main__":
    g = Grid()
    assert g.popcount() == 0

    g[0, 0] = 1
    assert g.popcount() == 1
