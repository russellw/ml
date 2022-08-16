import random

from etc import *
from simplify import simplify
import interpreter
import types1


def lam(env, t, depth):
    params = []
    paramts = []
    for paramt in t[2:]:
        params.append("x" + str(env.count()))
        paramts.append(paramt)
    env = Env(env, params, paramts)
    body = expr(env, t[1], depth)
    return "lambda", tuple(params), body


def expr(env, t, depth):
    t = freshVars(t)
    s = []

    # required or decided to return an atom
    if not depth or not random.randrange(0, 16):
        # available local variables that match the required type
        for x in env.keys1():
            if types1.unify({}, env.get(x), t):
                s.append(x)

        # some types can also be provided by literals
        match t:
            case "bool":
                s.append(False)
                s.append(True)
            case "num":
                s.append(0)
                s.append(1)
            case "fn", *_:
                # if we were supposed to be returning an atom, prefer to avoid further recursion,
                # but if the required return type is a function,
                # and we don't have any variables of that function type to hand,
                # then we don't have a choice
                if not s:
                    return lam(env, t, 0)
            case "list", _:
                s.append(())

        # choose a suitable atom at random
        return random.choice(s)

    # one more level of compound recursion
    depth -= 1

    # operators that match the required type
    for name, u, _ in interpreter.ops:
        if u and types1.unify({}, u[0], t):
            s.append((name, u))
    match t:
        case "fn", *_:
            s.append(("lambda", None))

    # choose a suitable operator at random
    name, u = random.choice(s)

    # recursively generate arguments
    if name == "lambda":
        return lam(env, t, depth)

    d = {}
    types1.unify(d, u[0], t)
    u = replace(d, u)

    s = [name]
    for t in u[1:]:
        s.append(expr(env, t, depth))
    return tuple(s)


def consistent(a, b, xs):
    env = Env()
    for x in xs:
        env["x"] = x
        y = interpreter.ev(env, a)
        z = interpreter.ev(env, b)
        if y != z:
            print(a)
            print(b)
            print(x)
            print(y)
            print(z)
            exit(1)


def trivial(a, xs):
    if not isinstance(a, tuple):
        return True
    env = Env()
    ys = set()
    for x in xs:
        env["x"] = x
        y = interpreter.ev(env, a)
        ys.add(y)
    return len(ys) == 1


if __name__ == "__main__":
    random.seed(0)
    env = Env()

    env["x"] = "num"
    for i in range(10000):
        try:
            a = expr(env, "num", 5)
            b = simplify(a)
            xs = range(10)
            consistent(a, b, xs)
            if trivial(b, xs):
                continue
            print(a)
            print(b)
            print()
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
