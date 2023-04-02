import inspect

symi = 0


def gensym():
    global symi
    a = "_" + str(symi)
    symi += 1
    return a


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {a}")


def tuple_depth(a):
    if isinstance(a, tuple):
        return max(map(tuple_depth, a), default=0) + 1
    return 0


def indent(n):
    for i in range(n):
        print("  ", end="")


def pprint1(a, dent):
    if isinstance(a, tuple):
        print("(", end="")
        if a:
            print(a[0], end="")
            if tuple_depth(a) <= 5:
                for b in a[1:]:
                    print(" ", end="")
                    pprint1(b, 0)
            else:
                dent += 1
                for b in a[1:]:
                    print()
                    indent(dent)
                    pprint1(b, dent)
        print(")", end="")
        return
    print(a, end="")


def pprint(a):
    pprint1(a, 0)
    print()


if __name__ == "__main__":
    assert tuple_depth(5) == 0
    assert tuple_depth((5,)) == 1
    assert tuple_depth(()) == 1
    print("ok")
