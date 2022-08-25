import argparse
import random

from etc import *
import interpreter


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


def getLive(p):
    s = set()

    def rec(k):
        if k in s:
            return
        s.add(k)
        for a in p[k]:
            if a in p:
                rec(a)

    rec("a")
    return s


def rmDead(p):
    q = {}
    for k in sorted(getLive(p)):
        q[k] = p[k]
    return q


if __name__ == "__main__":
    assert fname(0) == "a"
    assert fname(1) == "b"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", action="store_true", help="args are bit strings instead of numbers"
    )
    parser.add_argument(
        "-s", metavar="seed", help="random seed, default is current time"
    )
    args = parser.parse_args()

    random.seed(args.s)

    xs = range(10)
    if args.b:
        xs = []
        for i in range(10):
            xs.append(tuple(random.randrange(2) for j in range(10)))

    iterations = 1000000
    interval = iterations // 10
    ps = []
    for iteration in range(iterations):
        if iteration % interval == 0:
            print(iteration)
        try:
            p = mk(2, 10)
            p = rmDead(p)
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
