import random

from interpreter import defs


def mk(depth):
    if depth == 0 or random.random() < 0.10:
        return random.choice((0, 1, (), "x"))
    o = random.choice(list(defs))
    v = [o]
    for i in range(defs[o].arity):
        v.append(mk(depth - 1))
    return tuple(v)
