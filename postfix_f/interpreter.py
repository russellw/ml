from etc import *


def sub():
    b = stack.pop()
    a = stack.pop()
    stack.append(a - b)


def pow1():
    b = stack.pop()
    if b > 1000:
        raise ValueError()
    a = stack.pop()
    stack.append(a ** b)


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
    stack[-1] = not stack[-1]


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


ops = {
    "add": add,
    "and": and1,
    "div": div,
    "dup": lambda: stack.append(stack[-1]),
    "eq": eq,
    "floordiv": floordiv,
    "le": le,
    "lt": lt,
    "mod": mod,
    "mul": mul,
    "not": not1,
    "one": lambda: stack.append(1),
    "or": or1,
    "pop": lambda: stack.pop(),
    "pow": pow1,
    "sub": sub,
    "swap": swap,
    "zero": lambda: stack.append(0),
}


def call(f):
    for a in f:
        g = program.get(a)
        if g is not None:
            call(g)
            continue
        ops[a]()


def run(p, x):
    global program
    global stack
    program = p
    stack = [x]
    call(p["a"])
    return stack[-1]


def good(p, xs):
    # a program is considered good for a set of inputs,
    # if it handles all the inputs without crashing,
    # and it is nontrivial i.e. does not return the same value for every input
    ys = set()
    for x in xs:
        y = run(p, x)
        ys.add(y)
    return len(ys) > 1


def test(f, x, y):
    p = {"a": f}
    assert run(p, x) == y


def test_good(f, xs):
    p = {"a": f}
    assert good(p, xs)


if __name__ == "__main__":
    test(("not",), 1, 0)
    test(("dup", "not"), 1, 0)
    test(("one", "one", "add"), 0, 2)
    test(("zero", "one", "sub"), 0, -1)

    xs = range(10)
    test_good(("dup",), xs)
    test_good(("dup", "not"), xs)

    print("ok")
