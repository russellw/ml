import random

from etc import *
from simplify import simplify
import interpreter

ops = []
arity = {}
for o, n, _ in interpreter.ops:
    ops.append(o)
    arity[o] = n


def expr(xdepth, depth):
    if not depth or not random.randrange(0, 16):
        s = [0, 1, (), "x"]
        for i in range(xdepth):
            s.append(f"x{i}")
        return random.choice(s)
    depth -= 1
    o = random.choice(ops)
    if o == "lambda":
        n = random.randint(1, 3)
        params = []
        for i in range(n):
            params.append(f"x{xdepth+i}")
        body = expr(xdepth + n, depth)
        return "lambda", tuple(params), body
    s = [o]
    for i in range(arity[o]):
        s.append(expr(xdepth, depth))
    return tuple(s)


def consistent(a, b, xs):
    for x in xs:
        y = interpreter.eval1(a, x)
        z = interpreter.eval1(b, x)
        if isinstance(y, interpreter.Closure) and isinstance(z, interpreter.Closure):
            raise TypeError()
        if y != z:
            print(a)
            print(b)
            print(x)
            print(y)
            print(z)
        assert y == z


def trivial(a, xs):
    if not isinstance(a, tuple):
        return 1
    ys = set()
    for x in xs:
        y = interpreter.eval1(a, x)
        ys.add(y)
    return len(ys) == 1


if __name__ == "__main__":
    random.seed(0)

    seen = set()
    for i in range(100000):
        try:
            a = expr(0, 5)
            b = simplify(a)
            xs = range(10)
            consistent(a, b, xs)
            if trivial(b, xs):
                continue
            if a in seen:
                continue
            seen.add(a)
            print(a)
            print(b)
            print()
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass

    print(len(seen))
