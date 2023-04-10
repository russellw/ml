import random


class Grid:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x1 = 0
        self.y1 = 0
        self.data = []
        self.check()

    def idx(self, x, y):
        x -= self.x
        y -= self.y
        height = self.y1 - self.y
        return y * height + x

    def __getitem__(self, x, y):
        if not (self.x <= x < self.x1) or not (self.y <= y < self.y1):
            return 0
        return self.data[self.idx(x, y)]

    def occupied_y(self, y):
        for x in range(self.x, self.x1):
            if self[x, y]:
                return True

    def occupied_x(self, x):
        for y in range(self.y, self.y1):
            if self[x, y]:
                return True

    def bound(self):
        x = self.x
        while x < self.x1 and not self.occupied_x(x):
            x += 1

        x1 = self.x1
        while x < x1 and not self.occupied_x(x1 - 1):
            x1 -= 1

        y = self.y
        while y < self.y1 and not self.occupied_y(y):
            y += 1

        y1 = self.y1
        while y < y1 and not self.occupied_y(y1 - 1):
            y1 -= 1

        return x, y, x1, y1

    def expand(self):
        width = self.x1 - self.x
        if occupied_y(self.y):
            self.y -= 1
            self.data = [0] * width + self.data
        if occupied_y(self.y1 - 1):
            self.y1 += 1
            self.data += [0] * width

        height = self.y1 - self.y
        if occupied_x(self.x):
            self.x -= 1
            self.data = [0] * height + self.data
        if occupied_y(self.y1 - 1):
            self.y1 += 1
            self.data += [0] * height

        self.check()

    def popcount(self):
        return sum(self.data)

    def run(self, steps=1):
        for step in range(steps):
            self.check()
            new = [0] * len(self.data)
            for y in range(self.y, self.y1):
                for x in range(self.x, self.x1):
                    n = 0
                    for y2 in range(y - 1, y + 2):
                        for x2 in range(x - 1, x + 2):
                            n += self[x2, y2]
                    if n == 3 or n == 2 and self[x, y]:
                        new[self.idx(x, y)] = 1
            self.data = new
            self.expand()

    def check(self):
        assert (self.x1 - self.x) * (self.y1 - self.y) == len(self.data)
        assert not self.occupied_x(self.x)
        assert not self.occupied_x(self.x1 - 1)
        assert not self.occupied_y(self.y)
        assert not self.occupied_y(self.y1 - 1)


def randgrid(size):
    rows = []
    for i in range(size):
        rows.append([random.randrange(2) for j in range(size)])
    return Grid(rows)


if __name__ == "__main__":
    g = Grid()
    assert g.popcount() == 0
