import random

from interpreter import defs
from unify import replace, unify


def is_fn(t):
    return isinstance(t, tuple) and t[0] == "fn"


def simplify(a):
    return a


def mk1(t, env, depth):
    # atom
    if depth == 0 or random.random() < 0.5:
        v = []
        for o in env:
            if not is_fn(env[o]) and unify(env[o], t, {}):
                v.append(o)
        if not v:
            raise Exception(t)
        return random.choice(v)

    # choose op
    v = []
    for o in env:
        if is_fn(env[o]) and unify(env[o], t, {}):
            v.append(o)
    if not v:
        raise Exception(t)
    o = random.choice(v)

    # arg types
    d = {}
    unify(env[o], t, d)

    # make subexpression
    v = [o]
    for u in replace(env[o], d)[1:]:
        v.append(mk1(u, env, depth - 1))
    return simplify(v)


def mk(t, env):
    for o in defs:
        assert o not in env
        d = defs[o]
        env[o] = d.t
    return mk1(t, env, 3)


if __name__ == "__main__":
    random.seed(0)

    for i in range(50):
        try:
            print(mk("num", {"x": "num"}))
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
