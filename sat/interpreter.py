import operator

# https://docs.python.org/3/library/operator.html
ops = {
    "*": operator.mul,
    "+": operator.add,
    "-": operator.sub,
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "div": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "at": lambda a, b: a[b],
    "cons": lambda a, b: (a,) + b,
    "hd": lambda a: a[0],
    "len": lambda a: len(a),
    "map": lambda f, a: tuple(map(f, a)),
    "not": operator.not_,
    "tl": lambda a: a[1:],
    "/": operator.truediv,
}


def eva(a):
    if isinstance(a, tuple):
        f = eva(a[0])
        return f(*a[1:])
    if isinstance(a, str):
        return ops[a]
    return a


def test(a, b):
    assert eva(a) == b


if __name__ == "__main__":
    test(1, 1)
    test(("+", 1, 2), 3)
