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


class Var:
    # this class is intended for logic variables, not necessarily program variables
    def __init__(self, t=None):
        self.t = t

    def __repr__(self):
        if not hasattr(self, "name"):
            return "Var"
        return self.name


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def replace(d, a):
    if a in d:
        return replace(d, d[a])
    if isinstance(a, tuple):
        return tuple([replace(d, b) for b in a])
    return a


def simplify(a):
    if not isinstance(a, tuple):
        return a
    a = tuple(map(simplify, a))
    match a:
        case "+", x, 0:
            return x
        case "-", x, 0:
            return x
    return a
