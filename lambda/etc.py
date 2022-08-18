import inspect


def compose(a):
    s = []

    def rec(a):
        match a:
            case *_,:
                s.append("(")
                for b in a:
                    rec(b)
                s.append(")")
            case _:
                s.append(a)

    rec(a)
    return s


def size(a):
    match a:
        case *_,:
            return sum(map(size, a))
    return 1


def isConcrete(a):
    match a:
        case float() | int() | str():
            return 1
        case *_,:
            if len(a) >= 1000000:
                return
            return all(map(isConcrete, a))


def isConst(a):
    match a:
        case str():
            return
        case ():
            return 1
        case "quote", _:
            return 1
        case *_,:
            return
    return 1


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


if __name__ == "__main__":
    assert isConst(1)
    assert not isConst("a")
    assert isConst(())
    assert isConst(("quote", "a"))
    assert not isConst(("not", "a"))

    assert isConcrete(1)
    assert isConcrete(1.0)
    assert isConcrete(True)
    assert isConcrete("a")
    assert isConcrete((1, 2, 3))
    assert not isConcrete((1, 2, len))

    assert size(5) == 1
    assert size("abc") == 1
    assert size(["abc", "def"]) == 2

    assert compose(3) == [3]
    assert compose(("+", 3, "x")) == ["(", "+", 3, "x", ")"]

    print("ok")
