import argparse
import random

import interpreter
from gen import mk

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", help="expression depth", type=int, default=5)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10000)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()
print(args)

if args.seed is not None:
    random.seed(args.seed)


cache = set()

for i in range(args.epochs):
    program = mk(args.depth)
    cache.add(program)
print(len(cache))

u = set()
for a in cache:
    u.add(interpreter.simplify(a))
print(len(u))

rs = set()
for a in u:
    try:
        rs.add(interpreter.ev(a, {"x": 0}))
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        pass
print(rs)
print(len(rs))
