import operator
import random

# interpreter
def ev(env, a):
    if type(a) in (list, tuple):
        return evs[a[0]](env, a[1:])
    if type(a) is str:
        return env[a]
    return a


class Closure:
    def __init__(self, body, env):
        self.body = body
        self.env = env

    def __call__(self, arg):
        return ev(self.body, self.env + [arg])


ops = (
    ("*", ("num", "num", "num"), lambda env, a, b: ev(env, a) * ev(env, b)),
    ("+", ("num", "num", "num"), lambda env, a, b: ev(env, a) + ev(env, b)),
    ("-", ("num", "num", "num"), lambda env, a, b: ev(env, a) - ev(env, b)),
    ("/", ("num", "num", "num"), lambda env, a, b: ev(env, a) / ev(env, b)),
    ("<", ("bool", "num", "num"), lambda env, a, b: ev(env, a) < ev(env, b)),
    ("<=", ("bool", "num", "num"), lambda env, a, b: ev(env, a) <= ev(env, b)),
    ("==", ("bool", "T", "T"), lambda env, a, b: ev(env, a) == ev(env, b)),
    ("div", ("num", "num", "num"), lambda env, a, b: ev(env, a) // ev(env, b)),
    (
        "if",
        ("T", "bool", "T", "T"),
        lambda env, c, a, b: ev(env, a) if ev(env, c) else ev(env, b),
    ),
    ("lambda", None, lambda env, params, *body: Closure(env, params, body)),
    ("mod", ("num", "num", "num"), lambda env, a, b: ev(env, a) % ev(env, b)),
    ("pow", ("num", "num", "num"), lambda env, a, b: ev(env, a) ** ev(env, b)),
    ("at", ("T", ("list", "T"), "num"), lambda env, a, b: ev(env, a)[int(ev(env, b))]),
    (
        "cons",
        (("list", "T"), "T", ("list", "T")),
        lambda env, a, b: [ev(env, a)] + ev(env, b),
    ),
    ("car", ("T", ("list", "T")), lambda env, a: ev(env, a)[0]),
    ("len", ("num", ("list", "T")), lambda env, a: len(ev(env, a))),
    (
        "map",
        (("list", "T"), ("fn", "T", "T"), ("list", "T")),
        lambda env, a, b: [ev(env, a)] + ev(env, b),
    ),
    ("not", ("bool", "bool"), lambda env, a: not (ev(env, a))),
    ("cdr", (("list", "T"), ("list", "T")), lambda env, a: ev(env, a)[1:]),
)

evs = {}
for name, t, f in ops:
    evs[name] = f


# random generator
atoms = (0, 1, [], "arg")


def rand(depth):
    if not random.randrange(0, 16):
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


if __name__ == "__main__":
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
