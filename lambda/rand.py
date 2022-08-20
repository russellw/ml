import argparse
import random

from etc import *
import interpreter

ops = []
arity = {}
for o, n, _ in interpreter.ops:
    ops.append(o)
    arity[o] = n


def expr(depth, xdepth=1):
    if not depth or not random.randrange(0, 16):
        s = [0, 1, (), "x"]
        for i in range(1, xdepth):
            s.append(f"x{i}")
        return random.choice(s)
    depth -= 1
    o = random.choice(ops)
    if o == "lambda":
        n = random.randint(1, 3)
        params = []
        for i in range(n):
            params.append(f"x{xdepth+i}")
        body = expr(depth, xdepth + n)
        return "lambda", tuple(params), body
    s = [o]
    for i in range(arity[o]):
        s.append(expr(depth, xdepth))
    return tuple(s)


def good(a, xs):
    ys = set()
    for x in xs:
        y = interpreter.ev(a, (x,))
        if not isConcrete(y):
            return
        ys.add(y)
    return len(ys) > 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        action="store_true",
        help="expression args are bit strings instead of numbers",
    )
    parser.add_argument(
        "-c",
        metavar="count",
        type=int,
        default=100,
        help="number of iterations x 1,000",
    )
    parser.add_argument(
        "-d", metavar="depth", type=int, default=5, help="depth of expressions"
    )
    parser.add_argument(
        "-s", metavar="seed", help="random seed, default is current time"
    )
    args = parser.parse_args()
    args.c *= 1000

    random.seed(args.s)

    xs = range(10)
    if args.b:
        xs = []
        for i in range(10):
            xs.append(tuple(random.randrange(2) for j in range(10)))

    interval = args.c / 10
    seen = set()
    for i in range(args.c):
        if i % interval == 0:
            print(i)
        try:
            a = expr(args.d)
            a = deBruijn(a)
            if good(a, xs):
                seen.add(a)
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
    print(len(seen))
    n = 0
    for a in seen:
        n += size(a)
    print(n / len(seen))
