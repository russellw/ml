import random


def idx(self, x0, y0, x1, y1, x, y):
    x -= x0
    y -= y0
    width = x1 - x0
    return y * width + x


class Grid:
    def __init__(self):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.data = []
        self.check()

    def __getitem__(self, x, y):
        if not (self.x0 <= x < self.x1) or not (self.y0 <= y < self.y1):
            return 0
        return self.data[idx(self.x0, self.y0, self.x1, self.y1, x, y)]

    def occupied_y(self, y):
        for x in range(self.x0, self.x1):
            if self[x, y]:
                return True

    def occupied_x(self, x):
        for y in range(self.y0, self.y1):
            if self[x, y]:
                return True

    def bound(self):
        x0 = self.x0
        while x0 < self.x1 and not self.occupied_x(x0):
            x0 += 1

        x1 = self.x1
        while x0 < x1 and not self.occupied_x(x1 - 1):
            x1 -= 1

        y0 = self.y0
        while y0 < self.y1 and not self.occupied_y(y0):
            y0 += 1

        y1 = self.y1
        while y0 < y1 and not self.occupied_y(y1 - 1):
            y1 -= 1

        return x0, y0, x1, y1

    def expand(self):
        width = self.x1 - self.x0
        if occupied_y(self.y0):
            self.y0 -= 1
            self.data = [0] * width + self.data
        if occupied_y(self.y1 - 1):
            self.y1 += 1
            self.data += [0] * width

        height = self.y1 - self.y0
        if occupied_x(self.x0):
            self.x0 -= 1
            self.data = [0] * height + self.data
        if occupied_y(self.y1 - 1):
            self.y1 += 1
            self.data += [0] * height

        self.check()

    def popcount(self):
        return sum(self.data)

    def new_cell(self, x, y):
        n = 0
        for y2 in range(y - 1, y + 2):
            for x2 in range(x - 1, x + 2):
                if x2 or y2:
                    n += self[x2, y2]
        return n == 3 or n == 2 and self[x, y]

    def run(self, steps=1):
        for step in range(steps):
            self.check()

            # current size
            x0 = self.x0
            y0 = self.y0
            x1 = self.x1
            y1 = self.y1

            # expand north
            for x in range(self.x0 + 1, self.x1 - 1):
                if self.new_cell(x, self.y0 - 1):
                    y0 -= 1
                    break

            # expand south
            for x in range(self.x0 + 1, self.x1 - 1):
                if self.new_cell(x, self.y1):
                    y1 += 1
                    break

            # expand west
            for x in range(self.x0 + 1, self.x1 - 1):
                if self.new_cell(x, self.y0 - 1):
                    y0 -= 1
                    break

            new = [0] * len(self.data)
            for y in range(self.y0, self.y1):
                for x in range(self.x0, self.x1):
                    new[idx(self.x0, self.y0, self.x1, self.y1, x, y)] = self.new_cell(
                        x, y
                    )
            self.data = new
            self.expand()

    def check(self):
        assert (self.x1 - self.x0) * (self.y1 - self.y0) == len(self.data)


def randgrid(size):
    rows = []
    for i in range(size):
        rows.append([random.randrange(2) for j in range(size)])
    return Grid(rows)


if __name__ == "__main__":
    g = Grid()
    assert g.popcount() == 0

    g[0, 0] = 1
