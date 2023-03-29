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
    if isinstance(a, str):
        return env[a]
    if isinstance(a, tuple) and a:
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
    return a


def run(a):
    env = {}
    for o in ops:
        op = ops[o]
        if op.f:
            env[o] = op.f
    return ev(a, env)


def test(a, b):
    assert run(a) == b


if __name__ == "__main__":
    test(1, 1)
    test(("+", 1, 2), 3)
    print("ok")
