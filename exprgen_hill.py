import datetime
import logging
import math
import random
import sys
import time

random.seed(0)
start = time.time()

logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler(datetime.datetime.now().strftime("logs/%Y-%m-%d %H%M%S.log"))
)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def prn(a=""):
    logger.info(str(a))


def debug(a):
    logger.debug(str(a), stack_info=True)


prn(sys.argv)

ops = "+", "-", "*", "/", "sqrt"
leaves = 0.0, 1.0


def arity(o):
    if o == "sqrt":
        return 1
    return 2


def randcode(depth):
    if depth and random.random() < 0.8:
        o = random.choice(ops)
        return [o] + [randcode(depth - 1) for i in range(arity(o))]
    return random.choice(leaves)


def evaluate(a):
    if isinstance(a, list) or isinstance(a, tuple):
        try:
            a = list(map(evaluate, a))
            o = a[0]
            x = a[1]
            if o == "sqrt":
                return math.sqrt(x)
            y = a[2]
            return eval(f"x {o} y")
        except (ValueError, ZeroDivisionError):
            return 0.0
    return a


def size(a):
    if isinstance(a, list) or isinstance(a, tuple):
        n = 0
        for b in a:
            n += size(b)
        return n
    return 1


def loss(target, a):
    return (evaluate(a) - target) ** 2


def splice(a, i, b):
    if not i:
        return b
    assert isinstance(a, list) or isinstance(a, tuple)
    r = []
    for x in a:
        n = size(x)
        if 0 <= i < n:
            x = splice(x, i, b)
        i -= n
        r.append(x)
    return r


def mutate(a):
    i = random.randrange(size(a))
    return splice(a, i, randcode(5))


def hillclimb(target):
    a = randcode(5)
    best = a
    best_score = loss(target, a)
    i = 0
    print(i)
    print(best)
    print(evaluate(best))
    while best_score:
        i += 1
        b = mutate(a)
        score = loss(target, b)
        if score < best_score:
            best = b
            best_score = score
            print(i)
            print(best)
            print(evaluate(best))


prn()
for i in range(21):
    target = float(i)
    prn(target)
    hillclimb(target)
    prn()

seconds = time.time() - start
prn(f"{seconds:.3f} seconds")
prn(datetime.timedelta(seconds=seconds))
