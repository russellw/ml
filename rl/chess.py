# chess with adjustable board size
import random

size = 8

blank = None, 0
vals = {
    "p": 1,
    "n": 3,
    "b": 3,
    "r": 5,
    "q": 9,
}


class Board:
    def __init__(self):
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


def print_board(board):
    for i in range(size - 1, -1, -1):
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


def valid_moves(board):
    r = []

    def add(i1, j1):
        # outside the board
        if not (0 <= i1 < size and 0 <= j1 < size):
            return

        # onto own piece (including null move)
        p, color = board[i1, j1]
        if p and color == 0:
            return

        # valid move
        r.append((i, j, i1, j1))

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
    return r


board = Board()
print_board(board)
print(valid_moves(board))
