import argparse
import random

from etc import *
import interpreter

vocab = tuple(interpreter.ops.keys())


def mk(m, n):
    n = random.randint(m, n)
    return tuple(random.choice(vocab) for i in range(n))


if __name__ == "__main__":
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
    fs = []
    for i in range(args.c):
        if i % interval == 0:
            print(i)
        try:
            f = mk(args.m, args.n)
            if interpreter.good(f, xs):
                fs.append(f)
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
    print(len(fs))
    print(len(set(fs)))
    n = 0
    for f in fs:
        n += len(f)
    print(n / len(fs))
