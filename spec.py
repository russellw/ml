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


ops = "+", "-", "*", "//", "%", "==", "<", "<=", "not"
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
                return int(not x)
            y = evaluate(m, a[2])
            return int(eval(f"x {o} y"))
        except ZeroDivisionError:
            return 0
    if isinstance(a, str):
        return m[a]
    return a


ntests = 10


def accepts(spec, c):
    for x in range(ntests):
        m = {"x": x}
        m["y"] = evaluate(m, c)
        if not evaluate(m, spec):
            return False
    return True


candidates = [randcode(leaves, 5) for i in range(100)]


def difficulty(spec):
    for i in range(len(candidates)):
        c = candidates[i]
        if accepts(spec, c):
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


pr()
spec = randcode(leaves + ("y",), 5)
pr(spec)
pr(simplify(spec))
best = spec
best_score = difficulty(spec)
pr(best_score)
pr()
for i in range(1000):
    spec = mutate_spec(best)
    score = difficulty(spec)
    if score > best_score:
        pr(spec)
        pr(simplify(spec))
        pr(score)
        for c in candidates:
            if accepts(spec, c):
                pr(c)
                for x in range(ntests):
                    m = {"x": x}
                    m["y"] = evaluate(m, c)
                    pr(f"{m}: {evaluate(m, spec)}")
                break
        else:
            raise Exception()
        pr()
        best = spec
        best_score = score


seconds = time.time() - start
pr(f"{seconds:.3f} seconds")
pr(datetime.timedelta(seconds=seconds))
