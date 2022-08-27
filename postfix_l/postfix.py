from etc import *

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
        raise TypeError()
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
        raise TypeError()
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
        raise TypeError()
    stack.append(len(s))


def hd():
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError()
    stack.append(s[0])


def tl():
    s = stack.pop()
    if not isinstance(s, tuple):
        raise TypeError()
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
        raise TypeError()
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
    a = stack.pop()
    s = stack.pop()
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
        dup()
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
    if type(a) != list:
        stack.append(a)
        return
    for b in a:
        if type(b) is str:
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


# compose value to list of strings
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


if __name__ == "__main__":
    assert compose(3) == ["{", "1", "1", "}"]
    assert compose(("+", 3, "x0")) == ["(", "+", "{", "1", "1", "}", "x0", ")"]

    print("ok")
