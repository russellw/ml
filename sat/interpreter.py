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


def ev(a, env):
    if a.o == "const":
        return a.val

    if a.o == "and":
        return ev(a.args[0], env) and ev(a.args[1], env)
    if a.o == "or":
        return ev(a.args[0], env) or ev(a.args[1], env)

    if a.o == "if":
        return ev(a.args[1], env) if ev(a.args[0], env) else ev(a.args[2], env)

    f = ops[a.o].f
    args = [ev(b, env) for b in a.args]
    return f(*args)


def test(a, env, b):
    assert ev(a, env) == b


class Node:
    def __init__(self, o, *args):
        self.o = o
        self.args = args


class Const:
    def __init__(self, val):
        self.o = "const"
        self.val = val


if __name__ == "__main__":
    test(Const(1), {}, 1)
    test(Node("+", Const(1), Const(2)), {}, 3)
    print("ok")
