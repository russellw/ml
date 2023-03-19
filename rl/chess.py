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

        v[0][size // 2 - 1] = "q", 0
        v[0][size // 2] = "k", 0
        v[1] = [("p", 0)] * size

        v[size - 2] = [("p", 1)] * size
        v[size - 1][size // 2 - 1] = "q", 1
        v[size - 1][size // 2] = "k", 1

        self.v = v

    def __getitem__(self, ij):
        i, j = ij
        return self.v[i][j]


def print_board(board):
    for i in range(size - 1, -1, -1):
        for j in range(size):
            p, side = board[i, j]
            if not p:
                print(".", end="")
            else:
                if side == 0:
                    p = p.upper()
                print(p, end="")
            print(" ", end="")
        print()


board = Board()
print_board(board)
