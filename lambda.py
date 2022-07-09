import random
import operator

atoms = (0, 1, "arg")
ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "if": None,
    "lambda": None,
}


def rand(depth):
    if depth:
        depth -= 1
        o = random.choice(list(ops.keys()))
        n = 2
        if o == "if":
            n = 3
        elif o == "lambda":
            n = 1
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


def apply(f, args):
    x = args[0]
    if type(f) is Closure:
        return eva(f.body, f.env)
    y = args[1]
    return eval(str(x) + f + str(y))


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
    for i in range(1000):
        a = rand(3)
        try:
            x = eva(a, []) + 0
            print(a)
            print(x)
            print()
        except:
            pass
