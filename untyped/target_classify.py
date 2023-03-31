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

targets = set()
while len(targets) < 1000:
    target = mk(args.depth)
    targets.add(target)


def score1(program, target):
    try:
        x = interpreter.ev(program, {"x": target})
        y = interpreter.ev(target, {"x": x})
        return y
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return 0


def score(program):
    r = 0
    for target in targets:
        if score1(program, target):
            r += 1
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
