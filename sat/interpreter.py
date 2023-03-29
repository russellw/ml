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
    "nil": Def(("list", "$t"), ()),
    "*": Def(("fn", "num", "num", "num"), operator.mul),
    "neg": Def(("fn", "num", "num"), operator.neg),
    "+": Def(("fn", "num", "num", "num"), operator.add),
    "-": Def(("fn", "num", "num", "num"), operator.sub),
    "<": Def(("fn", "bool", "num", "num"), operator.lt),
    "<=": Def(("fn", "bool", "num", "num"), operator.le),
    "=": Def(("fn", "bool", "$t", "$t"), operator.eq),
    "div": Def(("fn", "num", "num", "num"), operator.floordiv),
    "mod": Def(("fn", "num", "num", "num"), operator.mod),
    "pow": Def(("fn", "num", "num", "num"), operator.pow),
    "at": Def(("fn", "$t", ("list", "$t"), "num"), lambda a, b: a[int(b)]),
    "cons": Def(("fn", ("list", "$t"), "$t", ("list", "$t")), lambda a, b: (a,) + b),
    "hd": Def(("fn", "$t", ("list", "$t")), lambda a: a[0]),
    "len": Def(("fn", "num", ("list", "$t")), lambda a: len(a)),
    # "map": Def((),lambda f, a: tuple(map(f, a))),
    "and": Def(("fn", "bool", "bool", "bool"), None),
    "or": Def(("fn", "bool", "bool", "bool"), None),
    "if": Def(("fn", "$t", "bool", "$t", "$t"), None),
    "not": Def(("fn", "bool", "bool"), operator.not_),
    "tl": Def(("fn", ("list", "$t"), ("list", "$t")), lambda a: a[1:]),
    "/": Def(("fn", "num", "num", "num"), operator.truediv),
}


def ev(a, env):
    # atom
    if isinstance(a, str):
        return env[a]

    # compound
    assert isinstance(a, tuple)
    o = a[0]

    # special form
    if o == "and":
        return ev(a[1], env) and ev(a[2], env)
    if o == "if":
        return ev(a[2], env) if ev(a[1], env) else ev(a[3], env)
    if o == "or":
        return ev(a[1], env) or ev(a[2], env)

    # call
    f = ev(o, env)
    args = [ev(b, env) for b in a[1:]]
    return f(*args)


def run(a, env):
    for o in defs:
        assert o not in env
        d = defs[o]
        if d.val is not None:
            env[o] = d.val
    return ev(a, env)


def test(a, b):
    assert run(a, {}) == b


if __name__ == "__main__":
    test("1", 1)
    test(("+", "1", "1"), 2)
    print("ok")
