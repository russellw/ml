import argparse
import random

from etc import *
import interpreter


def fname(i):
    return chr(ord("a") + i)


fcount = 10
vocab = tuple(interpreter.ops.keys()) + tuple(map(fname, range(fcount)))


def mkf(m, n):
    n = random.randint(m, n)
    return tuple(random.choice(vocab) for i in range(n))


def mk(m, n):
    p = {}
    for i in range(fcount):
        p[fname(i)] = mkf(m, n)
    return p


if __name__ == "__main__":
    assert fname(0) == "a"
    assert fname(1) == "b"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", action="store_true", help="args are bit strings instead of numbers"
    )
    parser.add_argument(
        "-c",
        metavar="count",
        type=int,
        default=1000,
        help="number of iterations x 1,000",
    )
    parser.add_argument("-m", type=int, default=2, help="min length")
    parser.add_argument("-n", type=int, default=10, help="max length")
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

    interval = args.c // 10
    ps = []
    for i in range(args.c):
        if i % interval == 0:
            print(i)
        try:
            p = mk(args.m, args.n)
            if interpreter.good(p, xs):
                ps.append(p)
        except (
            IndexError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

    # number of good programs
    print(len(ps))

    # number of distinct good programs
    s = set()
    for p in ps:
        s.add(frozenset(p.items()))
    print(len(s))

    # average program size
    n = 0
    for p in ps:
        for f in p.values():
            n += len(f)
    print(n / len(ps))
