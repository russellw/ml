import random

from unify import replace, unify

op_types = {
    "<": ("bool", "num", "num"),
    "<=": ("bool", "num", "num"),
    "not": ("bool", "bool"),
    "and": ("bool", "bool", "bool"),
    "or": ("bool", "bool", "bool"),
    "if": ("$t", "bool", "$t", "$t"),
    "=": ("bool", "$t", "$t"),
    "neg": ("num", "num"),
    "+": ("num", "num", "num"),
    "-": ("num", "num", "num"),
    "*": ("num", "num", "num"),
    "/": ("num", "num", "num"),
    "div": ("num", "num", "num"),
    "mod": ("num", "num", "num"),
    "pow": ("num", "num", "num"),
    "at": ("$t", ("list", "$t"), "num"),
    "hd": ("$t", ("list", "$t")),
    "tl": (("list", "$t"), ("list", "$t")),
}


class Node:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __repr__(self):
        return self.op + str(list(self.args))


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
        self.op = "const"
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


def mk(t, depth=3):
    # existing node
    r = []
    for a in nodes:
        if unify(a.t, t, {}):
            r.append(a)
    if r and (depth == 0 or random.random() < 0.5):
        return random.choice(r)

    # choose op
    ops = []
    for op in op_types:
        if unify(op_types[op][0], t, {}):
            ops.append(op)
    if not ops:
        raise Exception(t)
    op = random.choice(ops)

    # make subexpression
    d = {}
    unify(op_types[op], t, d)
    args = [mk(u, depth - 1) for u in replace(op_types[op], d)[1:]]
    return Node(op, *args)


if __name__ == "__main__":
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

    print("ok")
for i in range(10):
    print(mk("num"))
