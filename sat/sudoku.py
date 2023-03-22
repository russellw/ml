# https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
base = 3
side = base * base

# pattern for a baseline valid solution
def pattern(r, c):
    return (base * (r % base) + r // base + c) % side


# randomize rows, columns and numbers (of valid base pattern)
from random import sample


def shuffle(s):
    return sample(s, len(s))


rBase = range(base)
rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
nums = shuffle(range(1, base * base + 1))

# produce board using randomized baseline pattern
board = [[nums[pattern(r, c)] for c in cols] for r in rows]

squares = side * side
empties = squares * 3 // 4
for p in sample(range(squares), empties):
    board[p // side][p % side] = 0


# https://users.aalto.fi/~tjunttil/2020-DP-AUT/notes-sat/solving.html
D = base  # Subgrid dimension
N = D * D  # Grid dimension
clues = board
digits = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
}

# A helper: get the Dimacs CNF variable number for the variable v_{r,c,v}
# encoding the fact that the cell at (r,c) has the value v
def var(r, c, v):
    assert 1 <= r and r <= N and 1 <= c and c <= N and 1 <= v and v <= N
    return (r - 1) * N * N + (c - 1) * N + (v - 1) + 1


# Build the clauses in a list
cls = []  # The clauses: a list of integer lists
for r in range(1, N + 1):  # r runs over 1,...,N
    for c in range(1, N + 1):
        # The cell at (r,c) has at least one value
        cls.append([var(r, c, v) for v in range(1, N + 1)])
        # The cell at (r,c) has at most one value
        for v in range(1, N + 1):
            for w in range(v + 1, N + 1):
                cls.append([-var(r, c, v), -var(r, c, w)])
for v in range(1, N + 1):
    # Each row has the value v
    for r in range(1, N + 1):
        cls.append([var(r, c, v) for c in range(1, N + 1)])
    # Each column has the value v
    for c in range(1, N + 1):
        cls.append([var(r, c, v) for r in range(1, N + 1)])
    # Each subgrid has the value v
    for sr in range(0, D):
        for sc in range(0, D):
            cls.append(
                [
                    var(sr * D + rd, sc * D + cd, v)
                    for rd in range(1, D + 1)
                    for cd in range(1, D + 1)
                ]
            )
# The clues must be respected
for r in range(1, N + 1):
    for c in range(1, N + 1):
        if clues[r - 1][c - 1] in digits.keys():
            cls.append([var(r, c, digits[clues[r - 1][c - 1]])])

# Output the DIMACS CNF representation
# Print the header line
print("p cnf %d %d" % (N * N * N, len(cls)))
# Print the clauses
for c in cls:
    print(" ".join([str(l) for l in c]) + " 0")
