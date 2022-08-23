import argparse
import random

from etc import *
import interpreter
import rand

parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", action="store_true", help="args are bit strings instead of numbers"
)
parser.add_argument(
    "-c",
    metavar="count",
    type=int,
    default=1000,
    help="number of records",
)
parser.add_argument("-m", type=int, default=2, help="min length")
parser.add_argument("-n", type=int, default=10, help="max length")
parser.add_argument("-s", metavar="seed", help="random seed, default is current time")
args = parser.parse_args()

maxBits = args.n * bitLen(len(rand.vocab))
n = args.c
random.seed(args.s)

xs = range(10)
if args.b:
    xs = []
    for i in range(10):
        xs.append(tuple(random.randrange(2) for j in range(10)))


neg = []
pos = []
while len(neg) < n / 2 or len(pos) < n / 2:
    f = rand.mk(args.m, args.n)

    try:
        if not interpreter.good(f, xs):
            continue
        y = bool(interpreter.run(f, xs[1]))
    except (
        IndexError,
        OverflowError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        continue

    x = toBits(f, rand.vocab)
    x = fixLen(x, maxBits)
    r = x + (int(y),)

    if y:
        s = pos
    else:
        s = neg
    if len(s) < n / 2:
        s.append(r)
s = neg + pos
random.shuffle(s)
for r in s:
    print(",".join(str(a) for a in r))
