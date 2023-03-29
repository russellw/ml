import random

from interpreter import defs, run
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
        if is_fn(env[o]) and unify(env[o][1], t, {}):
            v.append(o)
    if not v:
        raise Exception(t)
    o = random.choice(v)

    # arg types
    d = {}
    unify(env[o][1], t, d)

    # make subexpression
    v = [o]
    for u in replace(env[o], d)[2:]:
        v.append(mk1(u, env, depth - 1))
    return simplify(tuple(v))


def mk(t, env):
    for o in defs:
        assert o not in env
        d = defs[o]
        env[o] = d.t
    return mk1(t, env, 3)


seen = set()


def print_new(a):
    if a not in seen:
        print(a)
        seen.add(a)


if __name__ == "__main__":
    random.seed(0)

    x = 10, 20, 30
    for i in range(1000):
        a = mk("num", {"x": ("list", "num")})
        print_new(a)
        try:
            if run(a, {"x": x}) == 30:
                print("***", i)
                break
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
