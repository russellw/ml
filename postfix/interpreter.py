from etc import *


def sub():
    b = stack.pop()
    a = stack.pop()
    stack.append(a - b)


def mul():
    b = stack.pop()
    a = stack.pop()
    stack.append(a * b)


def div():
    b = stack.pop()
    a = stack.pop()
    stack.append(a / b)


def floordiv():
    b = stack.pop()
    a = stack.pop()
    stack.append(a // b)


def mod():
    b = stack.pop()
    a = stack.pop()
    stack.append(a % b)


def eq():
    b = stack.pop()
    a = stack.pop()
    stack.append(a == b)


def add():
    b = stack.pop()
    a = stack.pop()
    stack.append(a + b)


def lt():
    b = stack.pop()
    a = stack.pop()
    stack.append(a < b)


def le():
    b = stack.pop()
    a = stack.pop()
    stack.append(a <= b)


def not1():
    a = stack.pop()
    stack.append(not a)


def and1():
    b = stack.pop()
    a = stack.pop()
    stack.append(a and b)


def or1():
    b = stack.pop()
    a = stack.pop()
    stack.append(a or b)


def dup():
    a = stack[-1]
    stack.append(a)


def pop():
    stack.pop()


def swap():
    b = stack.pop()
    a = stack.pop()
    stack.append(b)
    stack.append(a)


ops = {
    "%": mod,
    "*": mul,
    "+": add,
    "-": sub,
    "/": div,
    "//": floordiv,
    "<": lt,
    "<=": le,
    "=": eq,
    "and": and1,
    "dup": dup,
    "not": not1,
    "or": or1,
    "pop": pop,
    "swap": swap,
}


def call(f):
    for a in f:
        if type(a) is str:
            ops[a]()
            continue
        stack.append(a)


def run(f, x):
    global stack
    stack = [x]
    call(f)
    return stack[-1]


def test(f, x, y):
    assert run(f, x) == y


if __name__ == "__main__":
    test(("not",), 1, 0)
    test(
        (
            1,
            2,
            "+",
        ),
        0,
        3,
    )
    test(
        (
            1,
            2,
            "-",
        ),
        0,
        -1,
    )
    print("ok")
