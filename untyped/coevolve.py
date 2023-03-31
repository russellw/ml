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


def score_target(solvers, target):
    fail = 0
    succeed = 0
    for solver in solvers:
        if score(solver, target):
            succeed += 1
        else:
            fail += 1
    return fail * succeed


def score_solvers(solvers, targets):
    v = []
    for target in targets:
        v.append((score_solver(solver, targets), target))
    v.sort(key=lambda a: a[0], reverse=True)
    return v


def score_targets(solvers, targets):
    v = []
    for target in targets:
        v.append((score_target(solvers, target), target))
    v.sort(key=lambda a: a[0], reverse=True)
    return v


def improve_solvers(solvers, targets):
    v = score_solvers(solvers, targets)
    v = v[: len(v) * 100 // 90]
    v = [a[1] for a in v]
    v = set(v)
    while len(v) < 100:
        solver = mk(args.depth)
        v.add(solver)
    return v


solvers = set()
while len(solvers) < 100:
    solver = mk(args.depth)
    solvers.add(solver)

targets = set()
while len(targets) < 100:
    target = mk(args.depth)
    target = interpreter.simplify(target)
    targets.add(target)

for s, solver in score_solvers(solvers, targets)[:10]:
    print(solver)
print()

for i in range(1000):
    solvers = improve_solvers(solvers, targets)

for s, solver in score_solvers(solvers, targets)[:10]:
    print(solver)
print()

scores = score_targets(solvers, targets)

for s, target in scores[:10]:
    print(s, target)
print()

for s, target in scores[-3:]:
    print(s, target)
