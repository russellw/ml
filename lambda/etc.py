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


def const(a):
    match a:
        case str():
            return
        case Var():
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


def freeVars(a):
    free = set()

    def rec(bound, a):
        match a:
            case Var() as a:
                if a not in bound:
                    free.add(a)
            case "lambda", params, body:
                rec(bound | set(params), body)
            case _, *s:
                for a in s:
                    rec(bound, a)

    rec(set(), a)
    return free


def freshVars(a):
    d = {}

    def rec(a):
        match a:
            case Var() as a:
                if a not in d:
                    d[a] = Var(a.t)
            case _, *s:
                for a in s:
                    rec(a)

    rec(a)
    return replace(d, a)


def replace(d, a):
    if a in d:
        return replace(d, d[a])
    if isinstance(a, tuple):
        return tuple([replace(d, b) for b in a])
    return a


def simplify(a):
    # an atom is already in simplest form
    if not isinstance(a, tuple):
        return a

    # special form whose arguments cannot be recursively simplified
    match a:
        case "quote", x:
            if const(x):
                return x
            return a

    # recur on arguments
    a = tuple(map(simplify, a))

    # mathematical shortcuts
    match a:
        case "or", x, False:
            return x
        case "or", False, x:
            return x
        case "or", _, True:
            return True
        case "or", True, _:
            return True
        case "and", x, True:
            return x
        case "and", True, x:
            return x
        case "and", _, False:
            return False
        case "and", False, _:
            return False
        case "+", x, 0:
            return x
        case "+", 0, x:
            return x
        case "-", x, 0:
            return x
        case "-", x, y:
            if x == y:
                return 0
        case "*", _, 0:
            return 0
        case "*", 0, _:
            return 0
        case "*", x, 1:
            return x
        case "*", 1, x:
            return x
        case "/", x, 1:
            return x
        case "div", x, 1:
            return x
        case "if", True, x, _:
            return x
        case "if", False, _, x:
            return x
        case "if", _, x, y:
            if x == y:
                return x
        case "==", x, y:
            if x == y:
                return True
        case "<=", x, y:
            if x == y:
                return True
        case "<", x, y:
            if x == y:
                return False
        case "map", _, ():
            return ()
        case "map", ("lambda", (x,), y), s:
            if x == y:
                return s
        case "len", ("cons", _, ()):
            return 1
        case "len", ("cons", _, ("cons", _, ())):
            return 2
        case "len", ("cons", _, ("cons", _, ("cons", _, ()))):
            return 3

    # are all the arguments constant?
    if not all(map(const, a[1:])):
        return a

    # if so, we can evaluate the term immediately
    def unquote(a):
        match a:
            case "quote", x:
                return x
        return a

    def quote(a):
        if not const(a):
            return "quote", a
        return a

    match a:
        case "not", x:
            return eval(f"{a[0]} {x}")
        case (
            "and" | "or" | "==" | "<" | "<=" | "+" | "-" | "*" | "/" | "//" | "%" | "**"
        ), x, y:
            x = unquote(x)
            y = unquote(y)
            return eval(f"{x} {a[0]} {y}")
        case "cons", x, s:
            x = unquote(x)
            s = unquote(s)
            return quote((x,) + s)
        case "car", s:
            s = unquote(s)
            return quote(s[0])
        case "cdr", s:
            s = unquote(s)
            return quote(s[1:])
        case "len", s:
            s = unquote(s)
            return len(s)
    return a


if __name__ == "__main__":
    a = "a"
    x = Var()

    assert const(True)
    assert const(1)
    assert not const(a)
    assert not const(x)
    assert const(())
    assert const(("quote", "a"))
    assert not const(("not", "a"))

    assert freeVars("a") == set()
    assert freeVars(x) == set([x])
    assert freeVars(("+", x, x)) == set([x])

    print("ok")
