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
    body = rand(env, t[1], depth)
    return "lambda", tuple(params), body


def rand(env, t, depth):
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
        s.append(rand(env, t, depth))
    return tuple(s)


if __name__ == "__main__":
    random.seed(0)
    env = Env()
    n = 0

    env["x"] = "num"
    for i in range(100000):
        try:
            a = rand(env, "num", 5)
            b = simplify(a)
            if const(b):
                continue
            for x in range(10):
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
                n += 1
            print(a)
            print(b)
            print()
        except:
            pass

    print(n)
