import random

# Conway's Game of Life
def neighborhood8(i, j):
    for i1 in range(i - 1, i + 2):
        for j1 in range(j - 1, j + 2):
            if i1 == i and j1 == j:
                continue
            yield (i1, j1)


def blank_row(rows, i):
    for c in rows[i]:
        if c:
            return 0
    return 1


def blank_col(rows, j):
    for i in range(len(rows)):
        if rows[i][j]:
            return 0
    return 1


def trim(rows):
    # count blank south
    i2 = len(rows)
    while i2 and blank_row(rows, i2 - 1):
        i2 -= 1

    # count blank north
    i1 = 0
    while i1 < i2 and blank_row(rows, i1):
        i1 += 1

    # count blank east
    j2 = len(rows[0])
    while j2 and blank_col(rows, j2 - 1):
        j2 -= 1

    # count blank west
    j1 = 0
    while j1 < j2 and blank_col(rows, j1):
        j1 += 1

    # trim
    rows = rows[i1:i2]
    rows = [row[j1:j2] for row in rows]
    if not rows:
        rows = [[]]
    return rows, i1, j1


class Grid:
    def __init__(self, data="", origin=(0, 0)):
        rows = data
        if isinstance(data, str):
            data = data.strip()
            rows = []
            for s in data.split("\n"):
                rows.append([c != "." for c in s])
            width = max([len(row) for row in rows])
            for row in rows:
                row.extend([False] * (width - len(row)))
        rows, i1, j1 = trim(rows)
        self.rows = rows
        self.origin = origin[0] + i1, origin[1] + j1

    def __bool__(self):
        return self.popcount() > 0

    def __eq__(self, other):
        return self.origin == other.origin and self.rows == other.rows

    def __getitem__(self, key):
        i = key[0]
        if not (0 <= i < len(self.rows)):
            return False
        row = self.rows[i]
        j = key[1]
        if not (0 <= j < len(row)):
            return False
        return row[j]

    def __repr__(self):
        r = []
        if self.origin != (0, 0):
            r.append("(%d, %d)\n" % self.origin)
        for row in self.rows:
            for c in row:
                r.append("O" if c else ".")
                r.append(" ")
            r.append("\n")
        return "".join(r)

    def home(self):
        return Grid(self.rows)

    def popcount(self):
        n = 0
        for row in self.rows:
            n += sum(row)
        return n

    def step(self):
        height = len(self.rows)
        width = len(self.rows[0])
        rows = []
        for i in range(-1, height + 1):
            row = []
            for j in range(-1, width + 1):
                n = 0
                for i1, j1 in neighborhood8(i, j):
                    n += self[i1, j1]
                if self[i, j]:
                    c = n == 2 or n == 3
                else:
                    c = n == 3
                row.append(c)
            rows.append(row)
        return Grid(rows, (self.origin[0] - 1, self.origin[1] - 1))

    def steps(self, n):
        g = self
        for i in range(n):
            g = g.step()
        return g


gun = Grid(
    """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
"""
)


def randgrid(size):
    rows = []
    for i in range(size):
        rows.append([random.randrange(2) for j in range(size)])
    return Grid(rows)


def is_still(g):
    if not g:
        return
    h = step(g)
    return g == h


def is_blinker(g):
    if not g:
        return
    h = step(g)
    if g == h:
        return
    h = step(h)
    return g == h


def is_glider(g):
    if not g:
        return
    g = g.home()
    h = g
    for i in range(4):
        h = h.step()
        if g != h and g == h.home():
            return 1


# search
for i in range(1000000):
    g = randgrid(5)
    g = g.steps(10)
    if is_glider(g):
        print(g)
        print(i)
        break
