import argparse
import random

import interpreter
from gen import mk

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", help="expression depth", type=int, default=5)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10000)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

xs = []
while len(xs) < 100:
    x = []
    for i in range(random.randint(1, 5)):
        x.append(random.random())
    xs.append(tuple(x))


def max1(a):
    if len(a) == 1:
        return a[0]
    return max(*a)


def score1(program, x):
    try:
        if interpreter.ev(program, {"x": x}) == max1(x):
            return 1
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        pass
    return 0


def score(program):
    r = 0
    for x in xs:
        r += score1(program, x)
    return r


cache = set()

best_score = 0
for i in range(args.epochs):
    program = mk(args.depth)
    program = interpreter.simplify(program)

    # this only speed things up by a few percent
    if program in cache:
        continue
    cache.add(program)

    s = score(program)
    if s > best_score:
        print(i, s, program)
        best_score = s
