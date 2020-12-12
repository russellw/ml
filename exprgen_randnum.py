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


prn()
for i in range(11):
    target = float(i)
    prn(target)
    for j in range(1000000000):
        guess = randcode(5)
        if evaluate(guess) == target:
            prn(j)
            prn(guess)
            break
    prn()

seconds = time.time() - start
prn(f"{seconds:.3f} seconds")
prn(datetime.timedelta(seconds=seconds))
