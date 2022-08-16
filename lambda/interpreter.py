import random

from etc import *
import types1


def ev(env, a):
    if isinstance(a, tuple):
        if not a:
            return a
        o, *s = a
        return evs[o](env, *s)
    if isinstance(a, str):
        return env.get(a)
    return a


t = Var()
lst = ("list", t)
ops = (
    ("*", ("num", "num", "num"), lambda env, a, b: ev(env, a) * ev(env, b)),
    ("+", ("num", "num", "num"), lambda env, a, b: ev(env, a) + ev(env, b)),
    ("-", ("num", "num", "num"), lambda env, a, b: ev(env, a) - ev(env, b)),
    ("/", ("num", "num", "num"), lambda env, a, b: ev(env, a) / ev(env, b)),
    ("<", ("bool", "num", "num"), lambda env, a, b: ev(env, a) < ev(env, b)),
    ("<=", ("bool", "num", "num"), lambda env, a, b: ev(env, a) <= ev(env, b)),
    ("==", ("bool", t, t), lambda env, a, b: ev(env, a) == ev(env, b)),
    ("and", ("bool", "bool", "bool"), lambda env, a, b: ev(env, a) and ev(env, b)),
    ("at", (t, lst, "num"), lambda env, s, i: ev(env, s)[int(ev(env, i))]),
    ("call", None, lambda env, f, *s: ev(env, f)(*[ev(env, a) for a in s])),
    ("car", (t, lst), lambda env, s: ev(env, s)[0]),
    ("cdr", (lst, lst), lambda env, s: ev(env, s)[1:]),
    ("div", ("num", "num", "num"), lambda env, a, b: ev(env, a) // ev(env, b)),
    ("len", ("num", lst), lambda env, s: len(ev(env, s))),
    ("mod", ("num", "num", "num"), lambda env, a, b: ev(env, a) % ev(env, b)),
    ("not", ("bool", "bool"), lambda env, a: not (ev(env, a))),
    ("or", ("bool", "bool", "bool"), lambda env, a, b: ev(env, a) or ev(env, b)),
    ("pow", ("num", "num", "num"), lambda env, a, b: ev(env, a) ** ev(env, b)),
    (
        "cons",
        (lst, t, lst),
        lambda env, a, s: (ev(env, a),) + ev(env, s),
    ),
    (
        "if",
        (t, "bool", t, t),
        lambda env, c, a, b: ev(env, a) if ev(env, c) else ev(env, b),
    ),
    (
        "lambda",
        None,
        lambda env, params, body: lambda *args: ev(Env(env, params, args), body),
    ),
    (
        "map",
        (lst, ("fn", t, t), lst),
        lambda env, f, s: tuple(map(ev(env, f), ev(env, s))),
    ),
)

evs = {}
for name, _, f in ops:
    evs[name] = f


def lam(env, t, depth):
    params = []
    paramts = []
    for paramt in t[2:]:
        params.append("a" + str(env.count()))
        paramts.append(paramt)
    env = Env(env, params, paramts)
    body = rand(env, t[1], depth)
    return "lambda", tuple(params), body


def rand(env, t, depth):
    s = []

    # required or decided to return an atom
    if not depth or not random.randrange(0, 16):
        # available local variables that match the required type
        for a in env.keys1():
            if types1.unify({}, env.get(a), t):
                s.append(a)

        # some types can also be provided by literals
        match t:
            case "bool":
                s.append(False)
                s.append(True)
            case "num":
                s.append(0)
                s.append(1)
            case ("fn", *_):
                # if we were supposed to be returning an atom, prefer to avoid further recursion,
                # but if the required return type is a function,
                # and we don't have any variables of that function type to hand,
                # then we don't have a choice
                if not s:
                    return lam(env, t, 0)
            case ("list", _):
                s.append(())

        # choose a suitable atom at random
        return random.choice(s)

    # one more level of compound recursion
    depth -= 1

    # operators that match the required type
    for name, u, _ in ops:
        if u and types1.unify({}, u[0], t):
            s.append((name, u))
    match t:
        case ("fn", *_):
            s.append(("lambda", None))

    # choose a suitable operator at random
    name, u = random.choice(s)

    # recursively generate arguments
    if name == "lambda":
        return lam(env, t, depth)
    s = [name]
    for t in u[1:]:
        s.append(rand(env, t, depth))
    return tuple(s)


env = Env()
env["a"] = "num"
random.seed(1)
for i in range(20):
    print(rand(env, "num", 5))


def test(code, expected, arg=None):
    env = Env(None, ["a"], [arg])
    actual = ev(env, code)
    assert actual == expected


if __name__ == "__main__":
    test(2, 2)
    test("a", 3, 3)

    test(("+", "a", "a"), 6, 3)
    test(("*", 8, 3), 24)
    test(("/", 3, 4), 0.75)
    test(("div", 8, 4), 2)
    test(("mod", 10, 3), 1)
    test(("pow", 10, 3), 1000)

    test(("==", 3, 3), True)
    test(("==", 3, 4), False)

    test(("<", 1, 1), False)
    test(("<", 1, 2), True)
    test(("<", 2, 1), False)
    test(("<", 2, 2), False)

    test(("<=", 1, 1), True)
    test(("<=", 1, 2), True)
    test(("<=", 2, 1), False)
    test(("<=", 2, 2), True)

    test(("not", False), True)
    test(("not", True), False)

    test(("and", False, False), False)
    test(("and", False, True), False)
    test(("and", True, False), False)
    test(("and", True, True), True)
    test(("and", False, ("==", ("div", 1, 0), 99)), False)

    test(("or", False, False), False)
    test(("or", False, True), True)
    test(("or", True, False), True)
    test(("or", True, True), True)
    test(("or", True, ("==", ("div", 1, 0), 99)), True)

    test(("if", True, 1, ("div", 1, 0)), 1)
    test(("if", False, 1, 2), 2)

    test((), ())
    test(("cons", 1, ()), (1,))
    test(("cons", 1, ("cons", 2, ())), (1, 2))
    test(("car", "a"), 1, (1, 2, 3))
    test(("cdr", "a"), (2, 3), (1, 2, 3))
    test(("len", "a"), 3, (1, 2, 3))

    s = ("cons", 1, ("cons", 2, ("cons", 3, ())))
    test(("at", s, 0), 1)
    test(("at", s, 1), 2)
    test(("at", s, 2), 3)

    square = ("lambda", ("x",), ("*", "x", "x"))
    test(("call", square, ("+", 1, 2)), 9)
    test(("map", square, s), (1, 4, 9))

    print("ok")
    exit(0)
    for i in range(10000000):
        a = rand(4)
        try:
            x = ev(a, [])
            if len(x) < 2:
                continue
            print(a)
            print(x)
            print()
        except:
            pass
