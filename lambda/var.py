# this class is intended for logic variables, not necessarily program variables
class Var:
    def __init__(self, k):
        self.k = k

    def __eq__(self, other):
        return self.k == other.k

    def __hash__(self):
        return hash(self.k)

    def __repr__(self):
        return "$" + str(self.k)


if __name__ == "__main__":
    assert Var(1) == Var(1)
    assert Var(1) != Var(2)
    assert Var(1000000) == Var(1000000)
    assert Var(1000000) != Var(2000000)
    assert Var("a") == Var("a")
    assert Var("a") != Var("b")
