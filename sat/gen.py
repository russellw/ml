from unify import unify

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


def typeof(a):
    if isinstance(a, bool):
        return "bool"
    if isinstance(a, int) or isinstance(a, float):
        return "num"
    if isinstance(a, tuple):
        return "list", "$t"
    raise Exception(a)


def const(val):
    a = Node("const")
    a.val = val
    a.t = typeof(val)
    return a


nodes = [
    const(False),
    const(True),
    const(0),
    const(1),
    const(()),
]

if __name__ == "__main__":
    assert typeof(False) == "bool"
    assert typeof(True) == "bool"

    print("ok")
