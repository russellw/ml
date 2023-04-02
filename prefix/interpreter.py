import operator


class Def:
    def __init__(self, arity, val):
        self.arity = arity
        self.val = val


# https://docs.python.org/3/library/operator.html
defs = {
    "*": Def(2, operator.mul),
    "neg": Def(1, operator.neg),
    "+": Def(2, operator.add),
    "-": Def(2, operator.sub),
    "<": Def(2, operator.lt),
    "<=": Def(2, operator.le),
    "=": Def(2, operator.eq),
    "div": Def(2, operator.floordiv),
    "mod": Def(2, operator.mod),
    "pow": Def(2, operator.pow),
    "at": Def(2, lambda a, b: a[int(b)]),
    "cons": Def(2, lambda a, b: (a,) + b),
    "hd": Def(1, lambda a: a[0]),
    "len": Def(1, lambda a: len(a)),
    "list?": Def(1, lambda a: isinstance(a, tuple)),
    "sym?": Def(1, lambda a: isinstance(a, str)),
    "int?": Def(1, lambda a: isinstance(a, int)),
    "float?": Def(1, lambda a: isinstance(a, float)),
    "and": Def(2, None),
    "quote": Def(1, None),
    "or": Def(2, None),
    "if": Def(3, None),
    "not": Def(1, operator.not_),
    "tl": Def(1, lambda a: a[1:]),
    "/": Def(2, operator.truediv),
}


def ev(a, env):
    if isinstance(a, str):
        if a in env:
            return env[a]

        r = defs[a].val
        if r is None:
            raise Exception(a)
        return r
    if isinstance(a, tuple):
        o = a[0]

        if o == "and":
            return ev(a[1], env) and ev(a[2], env)
        if o == "fn":
            name=a[1]
            params=a[2]
            body=a[3]
            def f(*args):
                env=env.copy()
                for key,val in zip(params,args):
                    env[key]=val
                return ev(body,env)
            return f
        if o == "if":
            return ev(a[2], env) if ev(a[1], env) else ev(a[3], env)
        if o == "or":
            return ev(a[1], env) or ev(a[2], env)
        if o == "quote":
            return a[1]

        f = ev(o, env)
        args = [ev(b, env) for b in a[1:]]
        return f(*args)
    return a


def test(a, b):
    assert ev(a, {}) == b


if __name__ == "__main__":
    test(1, 1)
    test(("+", 1, 1), 2)
    test(("quote", "a"), "a")

    print("ok")
