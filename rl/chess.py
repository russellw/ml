# partial implementation of chess with adjustable board size
# limits:
# check is not implemented; taking the King wins the game
# castling is not implemented
# en passant is not implemented
# the code is optimized for simplicity rather than performance
import argparse
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("-z", "--size", help="random number seed", type=int, default=8)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

size = args.size

blank = None, 0
vals = {
    "p": 1.0,
    "n": 3.0,
    "b": 3.5,
    "r": 5.0,
    "q": 9.0,
    "k": 1e6,
}


class Board:
    def __init__(self, v=None):
        if v:
            self.v = v
            return

        v = []
        for i in range(size):
            v.append([blank] * size)

        w = ["r"]
        while len(w) + 1 < size // 2 - 1:
            w.append("n")
        w.append("b")

        v[0][size // 2 - 1] = "q", 0
        v[0][size // 2] = "k", 0
        v[1] = [("p", 0)] * size

        v[size - 2] = [("p", 1)] * size
        v[size - 1][size // 2 - 1] = "q", 1
        v[size - 1][size // 2] = "k", 1

        j = 0
        for p in w:
            v[0][j] = p, 0
            v[0][size - 1 - j] = p, 0
            v[size - 1][j] = p, 1
            v[size - 1][size - 1 - j] = p, 1
            j += 1

        self.v = v

    def __getitem__(self, ij):
        i, j = ij
        return self.v[i][j]

    def flip(self):
        v = []
        for i in range(size - 1, -1, -1):
            r = []
            for j in range(size):
                p, color = self[i, j]
                r.append((p, 1 - color))
            v.append(r)
        return Board(v)

    def move(self, i, j, i1, j1):
        v = []
        for r in self.v:
            v.append(r.copy())

        p, color = v[i][j]
        v[i][j] = blank

        if p == "p" and i1 == size - 1:
            p = "q"

        v[i1][j1] = p, color
        return Board(v)


def print_row(flipped, i):
    if flipped:
        i = size - 1 - i
    print(i + 1, end="")


def print_column(j):
    print(chr(97 + j), end="")


def print_move(board, flipped, i, j, i1, j1):
    p, color = board[i, j]
    assert p
    if p != "p":
        print(p.upper(), end="")
    print_column(j)
    print_row(flipped, i)
    if board[i1, j1][0]:
        print("x", end="")
    else:
        print("-", end="")
    print_column(j1)
    print_row(flipped, i1)


def print_board(board):
    print("   ", end="")
    for j in range(size):
        print_column(j)
        print(" ", end="")
    print()
    for i in range(size - 1, -1, -1):
        print("%2d " % (i + 1), end="")
        for j in range(size):
            p, color = board[i, j]
            if not p:
                print(".", end="")
            else:
                if color == 0:
                    p = p.upper()
                print(p, end="")
            print(" ", end="")
        print()
    print()


def valid_moves(board):
    v = []

    def add(i1, j1):
        # outside the board
        if not (0 <= i1 < size and 0 <= j1 < size):
            return

        # onto own piece (including null move)
        p, color = board[i1, j1]
        if p and color == 0:
            return

        # valid move
        v.append((i, j, i1, j1))

    def rook():
        # north
        for i1 in range(i + 1, size):
            add(i1, j)
            if board[i1, j][0]:
                break

        # south
        for i1 in range(i - 1, -1, -1):
            add(i1, j)
            if board[i1, j][0]:
                break

        # east
        for j1 in range(j + 1, size):
            add(i, j1)
            if board[i, j1][0]:
                break

        # west
        for j1 in range(j - 1, -1, -1):
            add(i, j1)
            if board[i, j1][0]:
                break

    def bishop():
        # northeast
        i1 = i + 1
        j1 = j + 1
        while i1 < size and j1 < size:
            add(i1, j1)
            if board[i1, j1][0]:
                break
            i1 += 1
            j1 += 1

        # southeast
        i1 = i - 1
        j1 = j + 1
        while i1 >= 0 and j1 < size:
            add(i1, j1)
            if board[i1, j1][0]:
                break
            i1 -= 1
            j1 += 1

        # southwest
        i1 = i - 1
        j1 = j - 1
        while i1 >= 0 and j1 >= 0:
            add(i1, j1)
            if board[i1, j1][0]:
                break
            i1 -= 1
            j1 -= 1

        # northwest
        i1 = i + 1
        j1 = j - 1
        while i1 < size and j1 >= 0:
            add(i1, j1)
            if board[i1, j1][0]:
                break
            i1 += 1
            j1 -= 1

    for i in range(size):
        for j in range(size):
            p, color = board[i, j]

            # empty square
            if not p:
                continue

            # opponent piece
            if color:
                continue

            # own pieces
            if p == "p":
                if not board[i + 1, j][0]:
                    add(i + 1, j)
                if i == 1:
                    for i1 in range(i + 2, size // 2):
                        if board[i1, j][0]:
                            break
                        add(i1, j)
                continue
            if p == "n":
                add(i + 2, j + 1)
                add(i + 1, j + 2)
                add(i - 1, j + 2)
                add(i - 2, j + 1)
                add(i - 2, j - 1)
                add(i - 1, j - 2)
                add(i + 1, j - 2)
                add(i + 2, j - 1)
                continue
            if p == "b":
                bishop()
                continue
            if p == "r":
                rook()
                continue
            if p == "q":
                rook()
                bishop()
                continue
            if p == "k":
                for i1 in range(i - 1, i + 2):
                    for j1 in range(j - 1, j + 2):
                        add(i1, j1)
    return v


def live(board):
    kings = [0, 0]
    for i in range(size):
        for j in range(size):
            p, color = board[i, j]
            if p == "k":
                kings[color] = 1
    return kings[0] and kings[1]


def val(board):
    r = 0.0
    for i in range(size):
        for j in range(size):
            p, color = board[i, j]
            if p:
                if color:
                    r -= vals[p]
                else:
                    r += vals[p]
    return r


def play(board):
    assert live(board)
    best = []
    best_val = -math.inf
    for m in valid_moves(board):
        b = board.move(*m)
        v = val(b)
        if v > best_val:
            best = [m]
            best_val = v
        elif v == best_val:
            best.append(m)
    assert best
    return random.choice(best)


board = Board()
move = 0
while live(board):
    move += 1
    print(move, end=". ")

    m = play(board)
    print_move(board, 0, *m)
    board = board.move(*m)

    if live(board):
        board = board.flip()

        m = play(board)
        print(" ", end="")
        print_move(board, 1, *m)
        board = board.move(*m)

        board = board.flip()

    print()
    print_board(board)
