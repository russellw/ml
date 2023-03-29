import random

from interpreter import ev, ops
from unify import replace, unify


class Node:
    def __init__(self, o, *args):
        self.o = o
        self.args = args

    def __repr__(self):
        return self.o + str(list(self.args))


def typeof(a):
    if isinstance(a, bool):
        return "bool"
    if isinstance(a, int) or isinstance(a, float):
        return "num"
    if isinstance(a, tuple):
        return "list", "$t"
    raise Exception(a)


class Const:
    def __init__(self, val):
        self.o = "const"
        self.val = val
        self.t = typeof(val)

    def __repr__(self):
        return str(self.val)


nodes = [
    Const(False),
    Const(True),
    Const(0),
    Const(1),
    Const(()),
]


def simplify(a):
    if all([b.o == "const" for b in a.args]):
        return Const(ev(a))
    return a


def mk(t, depth=3):
    # existing node
    v = []
    for a in nodes:
        if unify(a.t, t, {}):
            v.append(a)
    if v and (depth == 0 or random.random() < 0.5):
        return random.choice(v)

    # choose op
    v = []
    for o in ops:
        if unify(ops[o].t[0], t, {}):
            v.append(o)
    if not v:
        raise Exception(t)
    o = random.choice(v)

    # make subexpression
    d = {}
    unify(ops[o].t, t, d)
    args = [mk(u, depth - 1) for u in replace(ops[o].t, d)[1:]]
    return simplify(Node(o, *args))


if __name__ == "__main__":
    random.seed(0)

    assert typeof(False) == "bool"
    assert typeof(True) == "bool"

    d = {}
    a = Const(1)
    b = Const(2)
    d[a] = 1
    d[b] = 2
    assert len(d) == 2
    assert a in d
    assert b in d
    assert d[a] == 1
    assert d[b] == 2

    for i in range(50):
        try:
            print(mk("num"))
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
