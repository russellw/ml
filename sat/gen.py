import random

from interpreter import defs, run
from unify import replace, unify


def typeof(a):
    if isinstance(a, bool):
        return "bool"
    if isinstance(a, int) or isinstance(a, float):
        return "num"
    if isinstance(a, tuple):
        return "list", "$t"
    raise Exception(a)


nodes = [
    False,
    True,
    0,
    1,
    (),
]


def is_const(a):
    if isinstance(a, str):
        return
    if isinstance(a, tuple):
        return a == ()
    return 1


def simplify(a):
    if all([is_const(b) for b in a.args]):
        return run(a)
    return a


def mk(t, depth=3):
    # existing node
    v = []
    for a in nodes:
        if unify(a.t, t, {}):
            v.append(a)
    if v and (depth == 0 or random.random() < 0.5):
        return random.choice(v)

    # choose op
    v = []
    for o in ops:
        if unify(ops[o].t[0], t, {}):
            v.append(o)
    if not v:
        raise Exception(t)
    o = random.choice(v)

    # make subexpression
    d = {}
    unify(ops[o].t, t, d)
    args = [mk(u, depth - 1) for u in replace(ops[o].t, d)[1:]]
    return simplify(Node(o, *args))


if __name__ == "__main__":
    random.seed(0)

    assert typeof(False) == "bool"
    assert typeof(True) == "bool"

    for i in range(50):
        try:
            print(mk("num"))
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
