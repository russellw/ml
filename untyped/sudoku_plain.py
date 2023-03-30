# https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
base = 4
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


def expandLine(line):
    return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]


line0 = expandLine("╔═══╤═══╦═══╗")
line1 = expandLine("║ . │ . ║ . ║")
line2 = expandLine("╟───┼───╫───╢")
line3 = expandLine("╠═══╪═══╬═══╣")
line4 = expandLine("╚═══╧═══╩═══╝")

symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
nums = [[""] + [symbol[n] for n in row] for row in board]
print(line0)
for r in range(1, side + 1):
    print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
    print([line2, line3, line4][(r % side == 0) + (r % base == 0)])
