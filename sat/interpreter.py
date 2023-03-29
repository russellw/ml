import operator


class Def:
    def __init__(self, t, val):
        self.t = t
        self.val = val


# https://docs.python.org/3/library/operator.html
defs = {
    "0": Def("num", 0),
    "1": Def("num", 1),
    "false": Def("bool", False),
    "true": Def("bool", True),
    "*": Def(("num", "num", "num"), operator.mul),
    "neg": Def(("num", "num"), operator.neg),
    "+": Def(("num", "num", "num"), operator.add),
    "-": Def(("num", "num", "num"), operator.sub),
    "<": Def(("bool", "num", "num"), operator.lt),
    "<=": Def(("bool", "num", "num"), operator.le),
    "=": Def(("bool", "$t", "$t"), operator.eq),
    "div": Def(("num", "num", "num"), operator.floordiv),
    "mod": Def(("num", "num", "num"), operator.mod),
    "pow": Def(("num", "num", "num"), operator.pow),
    "at": Def(("$t", ("list", "$t"), "num"), lambda a, b: a[b]),
    "cons": Def((("list", "$t"), "$t", ("list", "$t")), lambda a, b: (a,) + b),
    "hd": Def(("$t", ("list", "$t")), lambda a: a[0]),
    "len": Def(("num", ("list", "$t")), lambda a: len(a)),
    # "map": Def((),lambda f, a: tuple(map(f, a))),
    "and": Def(("bool", "bool", "bool"), None),
    "or": Def(("bool", "bool", "bool"), None),
    "if": Def(("$t", "bool", "$t", "$t"), None),
    "not": Def(("bool", "bool"), operator.not_),
    "tl": Def((("list", "$t"), ("list", "$t")), lambda a: a[1:]),
    "/": Def(("num", "num", "num"), operator.truediv),
}


def ev(a, env):
    if isinstance(a, str):
        return env[a]
    assert isinstance(a, tuple) or isinstance(a, list)

    if not a:
        return ()
    o = a[0]

    if o == "and":
        return ev(a[1], env) and ev(a[2], env)
    if o == "if":
        return ev(a[2], env) if ev(a[1], env) else ev(a[3], env)
    if o == "or":
        return ev(a[1], env) or ev(a[2], env)

    f = ev(o, env)
    args = [ev(b, env) for b in a[1:]]
    return f(*args)


def run(a):
    env = {}
    for o in defs:
        d = defs[o]
        if d.val:
            env[o] = d.val
    return ev(a, env)


def test(a, b):
    assert run(a) == b


if __name__ == "__main__":
    test("1", 1)
    test(("+", "1", "1"), 2)
    print("ok")
