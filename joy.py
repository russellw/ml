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
    # stack
    "dup": dup,
    "pop": pop,
    "swap": swap,
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
    for a in code:
        if type(a) is str:
            ops[a]()
            continue
        stack.append(a)
    return stack[-1]


def test(s, r):
    v = lex(s)
    code = parse(v)
    x = run(code)
    assert x == r


if __name__ == "__main__":
    assert lex("3 dup *") == ["3", "dup", "*"]
    assert parse(lex("3 dup *")) == [3, "dup", "*"]
    test("3 dup *", 9)
    code = [3, "dup", "*"]
    x = run(code)
    print(x)
    exit(0)
    for i in range(20):
        code = rand(10)
        print(code)
        a = parse(code)
