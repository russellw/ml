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


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


if __name__ == "__main__":
    assert size(5) == 1
    assert size("abc") == 1
    assert size(("abc", "def")) == 2

    assert compose(3) == [3]
    assert compose(("+", 3, "x")) == ["(", "+", 3, "x", ")"]

    print("ok")
