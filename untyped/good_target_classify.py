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

solvers = set()
while len(solvers) < 100:
    solver = mk(args.depth)
    solvers.add(solver)

targets = set()
while len(targets) < 100:
    target = mk(args.depth)
    target = interpreter.simplify(target)
    targets.add(target)


def score1(solver, target):
    try:
        x = interpreter.ev(solver, {"x": target})
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return 0
    y = interpreter.ev(target, {"x": x})
    return bool(y)


def score(target):
    fail = 0
    succeed = 0
    for solver in solvers:
        try:
            if score1(solver, target):
                succeed += 1
            else:
                fail += 1
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
    return fail * succeed


scores = []
for target in targets:
    scores.append((score(target), target))
scores.sort(key=lambda a: a[0])

for s, target in scores[:3]:
    print(s, target)
print()
for s, target in scores[-3:]:
    print(s, target)
