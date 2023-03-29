import argparse
import random

from gen import mk
from interpreter import run

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", help="expression depth", type=int, default=3)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

xs = []
while len(xs) < 1000:
    x = []
    for i in range(random.randint(1, 10)):
        x.append(random.random())
    xs.append(tuple(x))


def max1(a):
    if len(a) == 1:
        return a[0]
    return max(*a)


def score1(program, x):
    try:
        if run(program, {"x": x}) == max1(x):
            return 1
    except (IndexError, ZeroDivisionError):
        pass
    return 0


def score(program):
    r = 0
    for x in xs:
        r += score1(program, x)
    return r


best_score = 0
for i in range(100000):
    program = mk("num", {"x": ("list", "num")})
    s = score(program)
    if s > best_score:
        print(s, program)
        best_score = s
