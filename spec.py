import datetime
import logging
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


def pr(a=""):
    logger.info(str(a))


def debug(a):
    logger.debug(str(a), stack_info=True)


pr(sys.argv)


ops = "+", "-", "*", "//", "%", "==", "<", "<=", "and", "or", "not"
leaves = 0, 1, "x"


def arity(o):
    if o == "not":
        return 1
    return 2


def const(a):
    if isinstance(a, bool):
        return True
    if isinstance(a, int):
        return True


def randcode(leaves, depth):
    if depth and random.random() < 0.8:
        o = random.choice(ops)
        return [o] + [randcode(leaves, depth - 1) for i in range(arity(o))]
    return random.choice(leaves)


def simplify(a):
    if isinstance(a, list) or isinstance(a, tuple):
        try:
            o = a[0]
            x = simplify(a[1])
            if arity(o) == 1:
                if not const(x):
                    return [o, x]
                return int(not x)
            y = simplify(a[2])
            if not (const(x) or const(y)):
                return [o, x, y]
            if const(x) and const(y):
                return int(eval(f"x {o} y"))
            if o in ("+", "-"):
                if x == 0:
                    return y
                if y == 0:
                    return x
            elif o == "*":
                if x == 0 or y == 0:
                    return 0
                if x == 1:
                    return y
                if y == 1:
                    return x
            elif o in ("//", "%"):
                if y == 0:
                    return 0
                if y == 1:
                    return x
            elif o == "and":
                if x == 0 or y == 0:
                    return 0
                if const(x) and x:
                    return y
            elif o == "or":
                if x == 0:
                    return y
                if y == 0:
                    return x
                if const(x) and x:
                    return x
            return [o, x, y]
        except ZeroDivisionError:
            return 0
    return a


def evaluate(m, a):
    if isinstance(a, list) or isinstance(a, tuple):
        try:
            o = a[0]
            x = evaluate(m, a[1])
            if o == "not":
                return not x
            y = evaluate(m, a[2])
            return eval(f"x {o} y")
        except ZeroDivisionError:
            return 0
    if isinstance(a, str):
        return m[a]
    return a

candidates=[randcode(leaves, 5)for i in range(100)]
ntests=10

def difficulty(spec):
    for c in candidates:
        for x in range(ntests):
            m = {"x": x}
            m["y"] = evaluate(m, c)
            if not evaluate(m, spec):
                break
        else:
            return i
    return -1


def size(a):
    if isinstance(a, list) or isinstance(a, tuple):
        n = 0
        for b in a:
            n += size(b)
        return n
    return 1


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


def mutate(leaves, a):
    i = random.randrange(size(a))
    return splice(a, i, randcode(leaves, 5))


def mutate_spec(a):
    return mutate(leaves + ("y",), a)


def hillclimb(a):
    pr(a)
    best = a
    best_score = difficulty(a)
    for i in range(100):
        b = mutate_spec(a)
        score = difficulty(b)
        if score > best_score:
            pr(b)
            pr(simplify(b))
            for c in candidates:
                pr(c)
                for x in range(ntests):
                    m = {"x": x}
                    m["y"] = evaluate(m, c)
                    pr(m)
                    pr(evaluate(m, spec))
            pr()
            best = b
            best_score = score


pr()
hillclimb(randcode(leaves + ("y",), 5), difficulty, mutate_spec)


seconds = time.time() - start
pr(f"{seconds:.3f} seconds")
pr(datetime.timedelta(seconds=seconds))
