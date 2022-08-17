import operator

from etc import *


class Closure:
    def __init__(self, env, params, body):
        self.env = env
        self.params = params
        self.body = body

    def __call__(self, *args):
        env = Env(self.env, self.params, args)
        return ev(env, self.body)


ops = (
    ("!=", 2, operator.ne),
    ("%", 2, operator.mod),
    ("*", 2, operator.mul),
    ("+", 2, operator.add),
    ("-", 2, operator.sub),
    ("/", 2, operator.truediv),
    ("//", 2, operator.floordiv),
    ("<", 2, operator.lt),
    ("<=", 2, operator.le),
    ("==", 2, operator.eq),
    ("abs", 1, abs),
    ("all", 1, all),
    ("and", 2, None),
    ("any", 1, any),
    ("bool", 1, bool),
    ("contains", 2, operator.contains),
    ("countOf", 2, operator.countOf),
    ("getitem", 2, operator.getitem),
    ("if", 3, None),
    ("int", 1, int),
    ("lambda", None, None),
    ("len", 1, len),
    ("map", 2, lambda f, s: tuple(map(f, s))),
    ("max", 2, max),
    ("min", 2, min),
    ("neg", 1, operator.neg),
    ("not", 1, operator.not_),
    ("or", 2, None),
    ("pow", 2, pow),
    ("quote", 1, None),
    ("range", 2, range),
    ("round", 1, round),
    ("slice", 3, lambda s, i, j: s[int(i) : int(j)]),
    ("sum", 1, sum),
    ("zip", 2, zip),
)

genv = Env()
for o, _, f in ops:
    genv[o] = f


def ev(env, a):
    if isinstance(a, str):
        return env.get(a)
    if not isinstance(a, tuple):
        return a
    match a:
        case ():
            return a
        case "quote", x:
            return x
        case "and", x, y:
            return ev(env, x) and ev(env, y)
        case "or", x, y:
            return ev(env, x) or ev(env, y)
        case "if", c, x, y:
            return ev(env, x) if ev(env, c) else ev(env, y)
        case "lambda", params, body:
            return Closure(env, params, body)
    f, *s = a
    f = ev(env, f)
    s = [ev(env, a) for a in s]
    return f(*s)


def eval1(a, x):
    env = Env(genv, ["x"], [x])
    return ev(env, a)


def test(a, y, x=None):
    z = eval1(a, x)
    if y != z:
        print(a)
        print(x)
        print(y)
        print(z)
    assert y == z


if __name__ == "__main__":
    test(2, 2)
    test("x", 3, 3)

    test(("+", "x", "x"), 6, 3)
    test(("*", 8, 3), 24)
    test(("/", 3, 4), 0.75)
    test(("//", 8, 4), 2)
    test(("%", 10, 3), 1)
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
    test(("and", False, ("==", ("//", 1, 0), 99)), False)

    test(("or", False, False), False)
    test(("or", False, True), True)
    test(("or", True, False), True)
    test(("or", True, True), True)
    test(("or", True, ("==", ("//", 1, 0), 99)), True)

    test(("if", True, 1, ("//", 1, 0)), 1)
    test(("if", False, 1, 2), 2)

    test((), ())
    test(("len", "x"), 3, (1, 2, 3))

    s = "quote", (1, 2, 3)
    test(("getitem", s, 0), 1)
    test(("getitem", s, 1), 2)
    test(("getitem", s, 2), 3)

    square = ("lambda", ("x",), ("*", "x", "x"))
    test((square, ("+", 1, 2)), 9)
    test(("map", square, s), (1, 4, 9))

    print("ok")
