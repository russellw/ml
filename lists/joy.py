# subset of Joy:
# https://en.wikipedia.org/wiki/Joy_(programming_language)
import operator
import random

import balanced_dpv


def immutable(a):
    if type(a) is list:
        return tuple(map(immutable, a))
    return a


stack = []

# operators
def is_list():
    x = stack.pop()
    stack.append(type(x) == list)


def is_sym():
    x = stack.pop()
    stack.append(type(x) == str)


def eq():
    y = stack.pop()
    x = stack.pop()
    stack.append(x == y)


def lt():
    y = stack.pop()
    x = stack.pop()
    stack.append(x < y)


def le():
    y = stack.pop()
    x = stack.pop()
    stack.append(x <= y)


def not1():
    x = stack.pop()
    stack.append(not x)


def and1():
    # TODO: does this need short-circuit evaluation?
    y = stack.pop()
    x = stack.pop()
    stack.append(x and y)


def or1():
    y = stack.pop()
    x = stack.pop()
    stack.append(x or y)


def dup():
    x = stack[-1]
    stack.append(x)


def pop():
    stack.pop()


def swap():
    y = stack.pop()
    x = stack.pop()
    stack.append(y)
    stack.append(x)


def cons():
    v = stack.pop()
    x = stack.pop()
    stack.append([x] + v)


def hd():
    v = stack.pop()
    stack.append(v[0])


def tl():
    v = stack.pop()
    stack.append(v[1:])


def in1():
    v = stack.pop()
    x = stack.pop()
    stack.append(x in v)


def map1():
    f = stack.pop()
    v = stack.pop()
    r = []
    for x in v:
        stack.append(x)
        r.append(run1(f))
    stack.append(r)


def filter1():
    f = stack.pop()
    v = stack.pop()
    r = []
    for x in v:
        stack.append(x)
        if run1(f):
            r.append(x)
    stack.append(r)


def fold():
    f = stack.pop()
    a = stack.pop()
    v = stack.pop()
    stack.append(a)
    for x in v:
        stack.append(x)
        run0(f)


def ii():
    f = stack.pop()
    run0(f)


def if1():
    y = stack.pop()
    x = stack.pop()
    c = stack.pop()
    if c:
        run0(x)
    else:
        run0(y)


def linrec1(c, then, rec1, rec2):
    dup()
    if run1(c):
        run0(then)
        return
    run0(rec1)
    linrec1(c, then, rec1, rec2)
    run0(rec2)


def linrec():
    rec2 = stack.pop()
    rec1 = stack.pop()
    then = stack.pop()
    c = stack.pop()
    linrec1(c, then, rec1, rec2)


ops = {
    # stack
    "dup": dup,
    "pop": pop,
    "swap": swap,
    # data types
    "list?": is_list,
    "sym?": is_sym,
    # comparison
    "=": eq,
    "<": lt,
    "<=": le,
    # logic
    "and": and1,
    "not": not1,
    "or": or1,
    # lists
    "cons": cons,
    "filter": filter1,
    "fold": fold,
    "hd": hd,
    "in": in1,
    "map": map1,
    "tl": tl,
    # control
    "i": ii,
    "if": if1,
    "linrec": linrec,
}


# parser
def constituent(c):
    if c.isspace():
        return
    if c in "()[]":
        return
    return 1


def lex(s):
    v = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()[]":
            v.append(c)
            i += 1
            continue
        j = i
        while i < len(s) and constituent(s[i]):
            i += 1
        v.append(s[j:i])
    return v


def parse(v):
    i = 0

    def expr():
        nonlocal i
        a = v[i]
        i += 1
        if a[0].isdigit():
            return int(a)
        if a == "False":
            return False
        if a == "True":
            return True
        if a not in ("(", "["):
            return a
        r = []
        while v[i] not in (")", "]"):
            r.append(expr())
        i += 1
        return r

    r = []
    while i < len(v) and v[i] not in (")", "]"):
        r.append(expr())
    i += 1
    return r


# interpreter
def run0(f):
    if type(f) != list:
        stack.append(f)
        return
    for a in f:
        if type(a) is str:
            ops[a]()
            continue
        stack.append(a)


def run1(f):
    run0(f)
    return stack.pop()


def run(f, arg):
    global stack
    stack = [arg]
    return run1(f)


# random generators
def rand_fn(n):
    alphabet = list(ops.keys()) + ["False", "True"]
    n = random.randrange(1, n)
    v = balanced_dpv.balanced_dp(n, alphabet).random()
    return parse(v)


def rand_input(n):
    v = []
    for i in range(n):
        v.append(random.randrange(2))
    return v


# top level
def test(s, r):
    v = lex(s)
    f = parse(v)
    x = run(f, 0)
    assert x == r


inputs = []
for i in range(20):
    inputs.append(rand_input(10))


def nontrivial(f):
    s = set()
    try:
        for x in inputs:
            y = run(f, x)
            s.add(immutable(y))
        return len(s) > 1
    except:
        pass


if __name__ == "__main__":
    assert lex("True dup and") == ["True", "dup", "and"]
    assert parse(lex("True dup and")) == [True, "dup", "and"]

    test("True dup and", True)

    fs = []
    while len(fs) < 5:
        f = rand_fn(10)
        if nontrivial(f):
            fs.append(f)
            x = inputs[0]
            y = run(f, x)
            print(f)
            print(x)
            print(y)
            print()
