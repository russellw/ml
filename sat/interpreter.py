import operator

# https://docs.python.org/3/library/operator.html
ops = {
    "*": operator.mul,
    "neg": operator.neg,
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


def ev(a):
    if a.op == "const":
        return a.val
    f = ops[a.op]
    args = [ev(b) for b in a.args]
    return f(*args)


def test(a, b):
    assert ev(a) == b


class Node:
    def __init__(self, op, *args):
        self.op = op
        self.args = args


def const(val):
    a = Node("const")
    a.val = val
    return a


if __name__ == "__main__":
    test(const(1), 1)
    test(Node("+", const(1), const(2)), 3)
    print("ok")
