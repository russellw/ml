import argparse
import random

import interpreter
from gen import mk

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", help="expression depth", type=int, default=8)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10000)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)


def run(program, x):
    try:
        return interpreter.ev(program, {"x": x})
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return 0


def score(solver, target):
    x = run(solver, target)
    y = run(target, x)
    return y


def score_solver(solver, targets):
    succeed = 0
    for target in targets:
        if score(solver, target):
            succeed += 1
    return succeed


targets = set()
while len(targets) < 10000:
    target = mk(args.depth)
    target = interpreter.simplify(target)
    targets.add(target)

for target in list(targets)[:10]:
    print(target)
print()

best_score = -1
for i in range(args.epochs):
    solver = mk(args.depth)
    solver = interpreter.simplify(solver)
    s = score_solver(solver, targets)
    if s > best_score:
        best_score = s
        print(i, s, solver)
