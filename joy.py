# subset of Joy:
# https://en.wikipedia.org/wiki/Joy_(programming_language)
import operator
import random

stack = []


# arithmetic
def add():
    y = stack.pop()
    x = stack.pop()
    stack.append(x + y)


def sub():
    y = stack.pop()
    x = stack.pop()
    stack.append(x - y)


def mul():
    y = stack.pop()
    x = stack.pop()
    stack.append(x * y)


def div():
    y = stack.pop()
    x = stack.pop()
    stack.append(x * y)


def floordiv():
    y = stack.pop()
    x = stack.pop()
    stack.append(x // y)


def mod():
    y = stack.pop()
    x = stack.pop()
    stack.append(x % y)


# comparison
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


# logic
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


def if1():
    y = stack.pop()
    x = stack.pop()
    c = stack.pop()
    if c:
        run(x)
    else:
        run(y)


# stack
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


# lists
def identity():
    f = stack.pop()
    run(f)


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
        run(f)


# recursion
def linrec1(c, then, rec1, rec2):
    if run1(c):
        run(then)
        return
    run(rec1)
    linrec1(c, then, rec1, rec2)
    run(rec2)


def linrec():
    rec2 = stack.pop()
    rec1 = stack.pop()
    then = stack.pop()
    c = stack.pop()
    linrec1(c, then, rec1, rec2)


ops = {
    # arithmetic
    "+": add,
    "-": sub,
    "*": mul,
    "/": div,
    "div": floordiv,
    "mod": mod,
    # comparison
    "=": eq,
    "<": lt,
    "<=": le,
    # logic
    "not": not1,
    "and": and1,
    "or": or1,
    "if": if1,
    # stack
    "dup": dup,
    "pop": pop,
    "swap": swap,
    # lists
    "i": identity,
    "map": map1,
    "filter": filter1,
    "fold": fold,
    # recursion
    "linrec": linrec,
}


# random generator
def rand(size):
    code = []
    for i in range(size):
        a = random.choice(symbols)
        code.append(a)
    return code


# parser
def constituent(c):
    if c.isspace():
        return
    if c in "[]":
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
        if c in "[]":
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
        if a != "[":
            return a
        r = []
        while v[i] != "]":
            r.append(expr())
        i += 1
        return r

    r = []
    while i < len(v) and v[i] != "]":
        r.append(expr())
    i += 1
    return r


# interpreter
def run(code):
    if type(code) != list:
        stack.append(code)
        return
    for a in code:
        if type(a) is str:
            ops[a]()
            continue
        stack.append(a)


def run1(code):
    run(code)
    return stack.pop()


def test(s, r):
    global stack
    stack = []
    v = lex(s)
    code = parse(v)
    print(code)
    run(code)
    print(stack)
    x = stack[-1]
    assert x == r
    print()


if __name__ == "__main__":
    assert lex("3 dup *") == ["3", "dup", "*"]
    assert parse(lex("3 dup *")) == [3, "dup", "*"]

    test("3 dup *", 9)
    test("[1 2 3 4] [dup *] map", [1, 4, 9, 16])
    test("[1 2 3 4] [2 swap <] filter", [3, 4])
    test("[2 5 3] 0 [+] fold", 10)
    test("[2 5 3] 0 [dup * +] fold", 38)
    test("[1 1 1 + +] i", 3)
    test("1 2 3 if", 2)
    test("1 [ 1 1 +] [1 1 1 + +] if", 2)
    test("0 [ 1 1 +] [1 1 1 + +] if", 3)
    test("4 not 1 [dup 1 -] * linrec", 24)

    exit(0)
    for i in range(20):
        code = rand(10)
        print(code)
        a = parse(code)
