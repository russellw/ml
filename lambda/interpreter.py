import operator
import random

from env import Env


class Closure:
    def __init__(self, env, params, body):
        self.env = env
        self.params = params
        self.body = body

    def __call__(self, args):
        env = Env(self.env, self.params, args)
        return ev(self.body, self.env + [arg])


def ev(env, a):
    if isinstance(a, tuple):
        o, *s = a
        return evs[o](env, *s)
    if isinstance(a, str):
        return env.get(a)
    return a


ops = (
    ("*", ("num", "num", "num"), lambda env, a, b: ev(env, a) * ev(env, b)),
    ("+", ("num", "num", "num"), lambda env, a, b: ev(env, a) + ev(env, b)),
    ("-", ("num", "num", "num"), lambda env, a, b: ev(env, a) - ev(env, b)),
    ("/", ("num", "num", "num"), lambda env, a, b: ev(env, a) / ev(env, b)),
    ("<", ("bool", "num", "num"), lambda env, a, b: ev(env, a) < ev(env, b)),
    ("<=", ("bool", "num", "num"), lambda env, a, b: ev(env, a) <= ev(env, b)),
    ("==", ("bool", "T", "T"), lambda env, a, b: ev(env, a) == ev(env, b)),
    ("and", ("bool", "bool", "bool"), lambda env, a, b: ev(env, a) and ev(env, b)),
    ("at", ("T", ("list", "T"), "num"), lambda env, s, i: ev(env, s)[int(ev(env, i))]),
    ("call", None, lambda env, f, *s: ev(env, f)(*[ev(env, a) for a in s])),
    ("car", ("T", ("list", "T")), lambda env, s: ev(env, s)[0]),
    ("cdr", (("list", "T"), ("list", "T")), lambda env, s: ev(env, s)[1:]),
    ("div", ("num", "num", "num"), lambda env, a, b: ev(env, a) // ev(env, b)),
    ("lambda", None, lambda env, params, body: Closure(env, params, body)),
    ("len", ("num", ("list", "T")), lambda env, s: len(ev(env, s))),
    ("mod", ("num", "num", "num"), lambda env, a, b: ev(env, a) % ev(env, b)),
    ("not", ("bool", "bool"), lambda env, a: not (ev(env, a))),
    ("or", ("bool", "bool", "bool"), lambda env, a, b: ev(env, a) or ev(env, b)),
    ("pow", ("num", "num", "num"), lambda env, a, b: ev(env, a) ** ev(env, b)),
    (
        "cons",
        (("list", "T"), "T", ("list", "T")),
        lambda env, a, s: [ev(env, a)] + ev(env, s),
    ),
    (
        "if",
        ("T", "bool", "T", "T"),
        lambda env, c, a, b: ev(env, a) if ev(env, c) else ev(env, b),
    ),
    (
        "map",
        (("list", "T"), ("fn", "T", "T"), ("list", "T")),
        lambda env, f, s: map(ev(env, f), ev(env, s)),
    ),
)

evs = {}
for name, t, f in ops:
    evs[name] = f

globalEnv = Env(None, ["nil"], [[]])

# random generator
atoms = (0, 1, [], "arg")


def rand(env, t, depth):
    if not depth or not random.randrange(0, 16):
        depth = 0
    if depth:
        depth -= 1
        o = random.choice(list(ops.keys()))
        n = 2
        if o in arity:
            n = arity[o]
        v = [o]
        for i in range(n):
            v.append(rand(depth))
        return v
    a = random.choice(atoms)
    if a == "arg":
        return [a, random.randrange(0, 2)]
    return a


# top level
def test(code, expected, arg=None):
    env = Env(globalEnv, ["a"], [arg])
    actual = ev(env, code)
    assert actual == expected


if __name__ == "__main__":
    test(2, 2)
    test("a", 3, 3)
    test(("+", "a", "a"), 6, 3)
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
