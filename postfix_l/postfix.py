import random

from etc import *
import balanced_dpv

maxSize = 1000


def check(a):
    if -maxSize <= a <= maxSize:
        return
    raise OverflowError(a)


# primitive functions
def sub():
    b = stack.pop()
    a = stack.pop()
    a -= b
    check(a)
    stack.append(a)


def mul():
    b = stack.pop()
    a = stack.pop()
    if isinstance(a, str):
        raise TypeError(a)
    if isinstance(a, tuple):
        check(len(a) * b)
        a *= b
    else:
        a *= b
        check(a)
    stack.append(a)


def floordiv():
    b = stack.pop()
    a = stack.pop()
    a //= b
    stack.append(a)


def mod():
    b = stack.pop()
    a = stack.pop()
    a %= b
    stack.append(a)


def eq():
    b = stack.pop()
    a = stack.pop()
    stack.append(a == b)


def add():
    b = stack.pop()
    a = stack.pop()
    if isinstance(a, str):
        raise TypeError(a)
    if isinstance(a, tuple):
        check(len(a) + len(b))
        a += b
    else:
        a += b
        check(a)
    stack.append(a)


def lt():
    b = stack.pop()
    a = stack.pop()
    stack.append(a < b)


def le():
    b = stack.pop()
    a = stack.pop()
    stack.append(a <= b)


def and1():
    b = stack.pop()
    a = stack.pop()
    stack.append(a and b)


def or1():
    b = stack.pop()
    a = stack.pop()
    stack.append(a or b)


def swap():
    b = stack.pop()
    a = stack.pop()
    stack.append(b)
    stack.append(a)


def len1():
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError(s)
    stack.append(len(s))


def hd():
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError(s)
    stack.append(s[0])


def tl():
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError(s)
    stack.append(s[1:])


def cons():
    s = stack.pop()
    check(len(s) + 1)
    a = stack.pop()
    s = (a,) + s
    stack.append(s)


def at():
    i = stack.pop()
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError(s)
    stack.append(s[i])


def isNum():
    a = stack.pop()
    stack.append(isinstance(a, int) or isinstance(a, float))


def drop():
    n = stack.pop()
    s = stack.pop()
    stack.append(s[n:])


def take():
    n = stack.pop()
    s = stack.pop()
    stack.append(s[:n])


def in1():
    s = stack.pop()
    a = stack.pop()
    stack.append(a in s)


def map1():
    f = stack.pop()
    s = stack.pop()
    r = []
    for a in s:
        stack.append(a)
        r.append(eval1(f))
    stack.append(tuple(r))


def filter1():
    f = stack.pop()
    s = stack.pop()
    r = []
    for a in s:
        stack.append(a)
        if eval1(f):
            r.append(a)
    stack.append(tuple(r))


def fold():
    f = stack.pop()
    s = stack.pop()
    a = stack.pop()
    stack.append(a)
    for a in s:
        stack.append(a)
        ev(f)


def if1():
    b = stack.pop()
    a = stack.pop()
    c = stack.pop()
    if c:
        ev(a)
    else:
        ev(b)


def linrec():
    rec2 = stack.pop()
    rec1 = stack.pop()
    then = stack.pop()
    c = stack.pop()

    def rec():
        stack.append(stack[-1])
        if eval1(c):
            ev(then)
            return
        ev(rec1)
        rec()
        ev(rec2)

    rec()


ops = {
    "%": mod,
    "*": mul,
    "+": add,
    "-": sub,
    "//": floordiv,
    "<": lt,
    "<=": le,
    "=": eq,
    "and": and1,
    "at": at,
    "cons": cons,
    "drop": drop,
    "dup": lambda: stack.append(stack[-1]),
    "filter": filter1,
    "fold": fold,
    "hd": hd,
    "eval": lambda: ev(stack.pop()),
    "if": if1,
    "in": in1,
    "len": len1,
    "linrec": linrec,
    "list?": lambda: stack.append(isinstance(stack.pop(), tuple)),
    "map": map1,
    "nil": lambda: stack.append(()),
    "not": lambda: stack.append(not stack.pop()),
    "num?": isNum,
    "or": or1,
    "pop": lambda: stack.pop(),
    "swap": swap,
    "sym?": lambda: stack.append(isinstance(stack.pop(), str)),
    "take": take,
    "tl": tl,
}


# interpreter
def ev(a):
    if not isinstance(a, tuple):
        stack.append(a)
        return
    for b in a:
        if isinstance(b, str):
            ops[b]()
            continue
        stack.append(b)


def eval1(a):
    ev(a)
    return stack.pop()


def run(a, x):
    global stack
    stack = [x]
    return eval1(a)


# parse from list of tokens
inputVocab = list(ops.keys())
inputVocab.append(0)
inputVocab.append(1)


def parse(s):
    i = 0

    def expr():
        nonlocal i

        # next token
        a = s[i]
        i += 1

        # number
        if not isinstance(a, str):
            return a

        # string
        if a not in ("(", "["):
            return a

        # list
        r = []
        while i < len(s) and s[i] not in (")", "]"):
            r.append(expr())
        i += 1
        return tuple(r)

    r = []
    while i < len(s):
        r.append(expr())
    return tuple(r)


# generate a random program
def rand():
    n = random.randint(2, 10)
    s = balanced_dpv.balanced_dp(n, inputVocab).random()
    return parse(s)


# compose to list of tokens
vocab = list(ops.keys())
vocab.append("(")


def compose(a):
    s = []

    def rec(a):
        if isinstance(a, str):
            s.append(a)
            return
        if isinstance(a, tuple):
            s.append("(")
            for b in a:
                rec(b)
            s.append(")")
            return
        if isinstance(a, int):
            s.append("{")
            for c in format(a, "b"):
                s.append(c)
            s.append("}")
            return
        raise TypeError()

    rec(a)
    return s


# a program is considered good for a set of inputs,
# if it handles all the inputs without crashing,
# and it is nontrivial i.e. does not return the same value for every input
def good(a, xs):
    ys = set()
    for x in xs:
        y = run(a, x)
        ys.add(y)
    return len(ys) > 1


# unit tests
def test(a, x, y):
    y1 = run(a, x)
    if y != y1:
        print(a)
        print(x)
        print(stack)
        print(y)
        print(y1)
    assert y == y1


if __name__ == "__main__":
    random.seed(0)

    assert parse((3, "dup", "*")) == (3, "dup", "*")
    assert parse((3, "[", "dup", "]", "*")) == (3, ("dup",), "*")

    assert compose(3) == ["{", "1", "1", "}"]
    assert compose(("+", 3, "x0")) == ["(", "+", "{", "1", "1", "}", "x0", ")"]

    test((1, 10, "-"), 0, -9)
    test((20, 6, "//"), 0, 3)
    test((20, 6, "%"), 0, 2)
    test((3, "dup", "*"), 0, 9)
    test(("dup", "*"), 3, 9)
    test(("dup", "+"), 3, 6)
    test((("dup", "*"), "map"), (1, 2, 3, 4), (1, 4, 9, 16))
    test(((2, "swap", "<"), "filter"), (1, 2, 3, 4), (3, 4))
    test((0, (2, 5, 3), ("+",), "fold"), None, 10)
    test(
        (
            0,
            (2, 5, 3),
            (
                "dup",
                "*",
                "+",
            ),
            "fold",
        ),
        None,
        38,
    )
    test(((1, 1, 1, "+", "+"), "eval"), None, 3)
    test((1, 2, 3, "if"), None, 2)
    test((0, 2, 3, "if"), None, 3)
    test((0, (1, 0, "/"), (3,), "if"), None, 3)
    test((("not",), ("pop", 1), ("dup", 1, "-"), ("*",), "linrec"), 4, 24)
    test((("not",), ("pop", 1), ("dup", 1, "-"), ("*",), "linrec"), 5, 120)
    test(
        (0, 1, 2, 3, 4, (), "cons", "cons", "cons", "cons", "cons"),
        None,
        (0, 1, 2, 3, 4),
    )
    test(
        (0, 1, 2, 3, 4, (), "cons", "cons", "cons", "cons", "cons", 2, "take"),
        None,
        (0, 1),
    )
    test(
        (0, 1, 2, 3, 4, (), "cons", "cons", "cons", "cons", "cons", 2, "drop"),
        None,
        (2, 3, 4),
    )

    fac = (("not",), ("pop", 1), ("dup", 1, "-"), ("*",), "linrec")
    xs = range(5)
    assert good(fac, xs)

    xs = range(10)
    for i in range(1000):
        a = rand()
        try:
            good(a, xs)
        except (IndexError, TypeError, ZeroDivisionError):
            pass

    xs = []
    for i in range(10):
        xs.append(tuple(random.randrange(2) for j in range(10)))
    for i in range(1000):
        a = rand()
        try:
            good(a, xs)
        except (IndexError, RecursionError, TypeError, ZeroDivisionError):
            pass

    print("ok")
