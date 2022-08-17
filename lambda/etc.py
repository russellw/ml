import inspect


class Env(dict):
    def __init__(self, outer=None, params=(), args=()):
        self.outer = outer
        self.update(zip(params, args))

    def count(self):
        n = 0
        env = self
        while env:
            n += len(env)
            env = env.outer
        return n

    def get(self, k):
        env = self
        while env:
            if k in env:
                return env[k]
            env = env.outer
        raise ValueError(k)

    def keys1(self):
        s = set()
        env = self
        while env:
            s.update(env.keys())
            env = env.outer
        return s


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

    print("ok")
