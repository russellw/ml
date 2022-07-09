import operator
import random

atoms = (0, 1, "arg")
ops = {
    "*": operator.mul,
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "div": operator.floordiv,
    "if": None,
    "lambda": None,
    "mod": operator.mod,
    "pow": operator.pow,
    "at": lambda a, b: a[int(b)],
    "cons": lambda a, b: [a] + b,
    "hd": lambda a: a[0],
    "len": lambda a: len(a),
    "map": map,
    "not": operator.not_,
    "tl": lambda a: a[1:],
}
arity = {
    "hd": 1,
    "if": 3,
    "lambda": 1,
    "len": 1,
    "not": 1,
    "tl": 1,
}


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


class Closure:
    def __init__(self, body, env):
        self.body = body
        self.env = env

    def __call__(self, arg):
        return eva(self.body, self.env + [arg])


def eva(a, env):
    if type(a) is list:
        o = a[0]
        if o == "arg":
            return env[-1 - a[1]]
        if o == "if":
            if eva(a[1], env):
                i = 2
            else:
                i = 3
            return eva(a[i], env)
        if o == "lambda":
            return Closure(a[1], env)
        f = eva(o, env)
        args = [eva(x, env) for x in a[1:]]
        return f(*args)
    if type(a) is str:
        return ops[a]
    return a


if __name__ == "__main__":
    for i in range(10000000):
        a = rand(4)
        try:
            x = eva(a, []) + 0
            if x < 3:
                continue
            print(a)
            print(x)
            print()
        except:
            pass
