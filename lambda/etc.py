import inspect


def compose(a):
    s = []

    def rec(a):
        if not isinstance(a, tuple):
            s.append(a)
            return
        s.append("(")
        for b in a:
            rec(b)
        s.append(")")

    rec(a)
    return s


def size(a):
    if not isinstance(a, tuple):
        return 1
    return sum(map(size, a))


def deBruijn(a, env=("x",)):
    # local variable
    if a in env:
        return "arg", env.index(a)

    # atom
    if not isinstance(a, tuple) or not a:
        return a

    # lambda
    o = a[0]
    if o == "lambda":
        params = a[1]
        a = deBruijn(a[2], tuple(reversed(params)) + env)
        for x in params:
            a = o, a
        return a

    # function call or other special form
    return tuple(deBruijn(x, env) for x in a)


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


if __name__ == "__main__":
    assert size(5) == 1
    assert size("abc") == 1
    assert size(("abc", "def")) == 2

    assert compose(3) == [3]
    assert compose(("+", 3, "x")) == ["(", "+", 3, "x", ")"]

    assert deBruijn(("+", 3, "x")) == ("+", 3, ("arg", 0))
    assert deBruijn(("lambda", ("x0", "x1"), ("+", "x", ("*", "x0", "x1")))) == (
        "lambda",
        ("lambda", ("+", ("arg", 2), ("*", ("arg", 1), ("arg", 0)))),
    )

    print("ok")
