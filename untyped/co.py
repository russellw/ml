import argparse
import operator
import random

from etc import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", help="expression depth", type=int, default=8)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10000)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

# global definitions
class Def:
    def __init__(self, arity, val):
        self.arity = arity
        self.val = val


# https://docs.python.org/3/library/operator.html
defs = {
    "*": Def(2, operator.mul),
    "neg": Def(1, operator.neg),
    "+": Def(2, operator.add),
    "-": Def(2, operator.sub),
    "<": Def(2, operator.lt),
    "<=": Def(2, operator.le),
    "=": Def(2, operator.eq),
    "div": Def(2, operator.floordiv),
    "mod": Def(2, operator.mod),
    "pow": Def(2, operator.pow),
    "at": Def(2, lambda a, b: a[int(b)]),
    "cons": Def(2, lambda a, b: (a,) + b),
    "hd": Def(1, lambda a: a[0]),
    "len": Def(1, lambda a: len(a)),
    "list?": Def(1, lambda a: isinstance(a, tuple)),
    "sym?": Def(1, lambda a: isinstance(a, str)),
    "int?": Def(1, lambda a: isinstance(a, int)),
    "float?": Def(1, lambda a: isinstance(a, float)),
    "and": Def(2, None),
    "quote": Def(1, None),
    "or": Def(2, None),
    "if": Def(3, None),
    "not": Def(1, operator.not_),
    "tl": Def(1, lambda a: a[1:]),
    "/": Def(2, operator.truediv),
}

# interpreter
def ev(a, env):
    if isinstance(a, str):
        if a in env:
            return env[a]

        r = defs[a].val
        if r is None:
            raise Exception(a)
        return r
    if isinstance(a, tuple):
        o = a[0]

        if o == "and":
            return ev(a[1], env) and ev(a[2], env)
        if o == "if":
            return ev(a[2], env) if ev(a[1], env) else ev(a[3], env)
        if o == "or":
            return ev(a[1], env) or ev(a[2], env)
        if o == "quote":
            return a[1]

        f = ev(o, env)
        args = [ev(b, env) for b in a[1:]]
        return f(*args)
    return a


def run(program, x):
    try:
        return ev(program, {"x": x})
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return 0


# simplifier
def is_const(a):
    if isinstance(a, str):
        return
    if isinstance(a, tuple):
        return a[0] == "quote"
    return 1


def simplify(a):
    if isinstance(a, tuple):
        o = a[0]

        # different but equivalent terms are not equivalent when quoted
        if o == "quote":
            return a

        # recursively simplify arguments
        a = tuple(map(simplify, a))

        # if all the arguments are constant, evaluate immediately
        if all(map(is_const, a[1:])):
            a = ev(a, {})
            if isinstance(a, str) or isinstance(a, tuple):
                return "quote", a
            return a

        # simplify based on semantics of the operator
        if is_const(a[1]):
            x = ev(a[1], {})
            if o == "and":
                return a[1] if not x else a[2]
            if o == "if":
                return a[2] if x else a[3]
            if o == "or":
                return a[1] if x else a[2]
    return a


# generator
def mk(depth):
    if depth == 0 or random.random() < 0.10:
        return random.choice((0, 1, ("quote", ()), "x"))
    o = random.choice(list(defs))
    v = [o]
    for i in range(defs[o].arity):
        v.append(mk(depth - 1))
    return tuple(v)


def gen():
    while 1:
        try:
            a = mk(args.depth)
            a = simplify(a)
            if is_const(a):
                continue
            return a
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass


def gens(n):
    # use dict instead of set for deterministic order
    v = {}
    while len(v) < 1000:
        target = gen()
        v[target] = 1
    return list(v.keys())


# evaluator
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


targets = gens(1000)
for target in targets[:10]:
    pprint(target)
print()

best_score = -1
for i in range(args.epochs):
    solver = gen()
    s = score_solver(solver, targets)
    if s > best_score:
        best_score = s
        print(i, s, solver)
