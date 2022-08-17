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


def const(a):
    match a:
        case str():
            return
        case ():
            return True
        case "quote", _:
            return True
        case *_,:
            return
    return True


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


if __name__ == "__main__":
    assert const(1)
    assert not const("a")
    assert const(())
    assert const(("quote", "a"))
    assert not const(("not", "a"))

    print("ok")
