import operator


class Op:
    def __init__(self, t, f):
        self.t = t
        self.f = f


# https://docs.python.org/3/library/operator.html
ops = {
    "*": Op(("num", "num", "num"), operator.mul),
    "neg": Op(("num", "num"), operator.neg),
    "+": Op(("num", "num", "num"), operator.add),
    "-": Op(("num", "num", "num"), operator.sub),
    "<": Op(("bool", "num", "num"), operator.lt),
    "<=": Op(("bool", "num", "num"), operator.le),
    "=": Op(("bool", "$t", "$t"), operator.eq),
    "div": Op(("num", "num", "num"), operator.floordiv),
    "mod": Op(("num", "num", "num"), operator.mod),
    "pow": Op(("num", "num", "num"), operator.pow),
    "at": Op(("$t", ("list", "$t"), "num"), lambda a, b: a[b]),
    "cons": Op((("list", "$t"), "$t", ("list", "$t")), lambda a, b: (a,) + b),
    "hd": Op(("$t", ("list", "$t")), lambda a: a[0]),
    "len": Op(("num", ("list", "$t")), lambda a: len(a)),
    # "map": Op((),lambda f, a: tuple(map(f, a))),
    "and": Op(("bool", "bool", "bool"), None),
    "or": Op(("bool", "bool", "bool"), None),
    "if": Op(("$t", "bool", "$t", "$t"), None),
    "not": Op(("bool", "bool"), operator.not_),
    "tl": Op((("list", "$t"), ("list", "$t")), lambda a: a[1:]),
    "/": Op(("num", "num", "num"), operator.truediv),
}


def ev(a):
    if a.o == "const":
        return a.val

    if a.o == "and":
        return ev(a.args[0]) and ev(a.args[1])
    if a.o == "or":
        return ev(a.args[0]) or ev(a.args[1])

    if a.o == "if":
        return ev(a.args[1]) if ev(a.args[0]) else ev(a.args[2])

    f = ops[a.o].f
    args = [ev(b) for b in a.args]
    return f(*args)


def test(a, b):
    assert ev(a) == b


class Node:
    def __init__(self, o, *args):
        self.o = o
        self.args = args


def const(val):
    a = Node("const")
    a.val = val
    return a


if __name__ == "__main__":
    test(const(1), 1)
    test(Node("+", const(1), const(2)), 3)
    print("ok")
