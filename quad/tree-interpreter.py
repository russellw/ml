import inspect
import operator
import random


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {a}")


symi = 0


def gensym():
    global symi
    a = "_" + str(symi)
    symi += 1
    return a


# tokenizer
filename = ""
text = ""
ti = 0
line = 0
tok = None


def constituent(c):
    if c.isspace():
        return
    if c in "()[];{}":
        return
    return 1


def lex():
    global ti
    global tok
    global line
    while ti < len(text):
        i = ti

        # whitespace
        if text[ti].isspace():
            if text[ti] == "\n":
                line += 1
            ti += 1
            continue

        # comment
        if text[ti] == ";":
            while text[ti] != "\n":
                ti += 1
            continue
        if text[ti] == "{":
            lin = line
            while text[ti] != "}":
                if ti == len(text):
                    raise Exception("%s:%d: unclosed block comment" % (filename, lin))
                if text[ti] == "\n":
                    line += 1
                ti += 1
            ti += 1
            continue

        # number
        if text[ti].isdigit() or (text[ti] == "-" and text[ti + 1].isdigit()):
            ti += 1
            while text[ti].isalnum():
                ti += 1
            if text[ti] == ".":
                ti += 1
                while text[ti].isalnum():
                    ti += 1
            tok = text[i:ti]
            return

        # word
        if constituent(text[ti]):
            while constituent(text[ti]):
                ti += 1
            tok = text[i:ti]
            return

        # punctuation
        if text[ti] in "()":
            ti += 1
            tok = text[i:ti]
            return

        # none of the above
        raise Exception("%s:%d: stray '%c' in program" % (filename, line, text[ti]))
    tok = None


# parser
def eat(k):
    if tok == k:
        lex()
        return 1


def assoc(a):
    if len(a) <= 3:
        return a
    o = a[0]
    return (o, a[1], assoc(((o,) + a[2:])))


def pairwise(a):
    if len(a) <= 3:
        return a
    o = a[0]

    u = ["do"]
    b = [0]
    for i in range(1, len(a)):
        b.append(gensym())
        u.append(("=", b[i], a[i]))

    v = ["and"]
    for i in range(1, len(a) - 1):
        v.append((o, b[i], b[i + 1]))
    u.append(tuple(v))

    return tuple(u)


def expr():
    line1 = line
    if tok == "(":
        v = []
        lex()
        while not eat(")"):
            v.append(expr())
        a = tuple(v)

        if not a:
            return a
        if a[0] in ("+", "*", "and", "or"):
            return assoc(a)
        if a[0] in ("==", "/=", "<", "<=", ">", ">="):
            return pairwise(a)
        if a[0] == "assert":
            return (
                "if",
                a[1],
                0,
                ("err", ("quote", ("%s:%d: assert failed" % (filename, line1)))),
            )
        if a[0] == "when":
            return "if", a[1], (("do",) + a[1:]), 0

        return a
    if tok[0].isdigit() or (tok[0] == "-" and len(tok) > 1 and tok[1].isdigit()):
        a = int(tok)
        lex()
        return a
    s = tok
    lex()
    return s


def parse():
    global text
    global line
    global ti
    text = open(filename).read() + "\n"
    ti = 0
    line = 1
    lex()
    v = []
    while tok:
        v.append(expr())
    return v


# interpreter
def err(s):
    raise Exception(s)


def pr(a):
    print(a, end="")


def prn(a):
    print(a)


class Def:
    def __init__(self, arity, val):
        self.arity = arity
        self.val = val


# https://docs.python.org/3/library/operator.html
defs = {
    "*": Def(2, operator.mul),
    "neg": Def(1, operator.neg),
    "+": Def(2, operator.add),
    "-": Def(2, operator.sub),
    "<": Def(2, operator.lt),
    "<=": Def(2, operator.le),
    "==": Def(2, operator.eq),
    "div": Def(2, operator.floordiv),
    "mod": Def(2, operator.mod),
    "pow": Def(2, operator.pow),
    "at": Def(2, lambda a, b: a[int(b)]),
    "cons": Def(2, lambda a, b: (a,) + b),
    "hd": Def(1, lambda a: a[0]),
    "len": Def(1, lambda a: len(a)),
    "list?": Def(1, lambda a: isinstance(a, tuple)),
    "sym?": Def(1, lambda a: isinstance(a, str)),
    "int?": Def(1, lambda a: isinstance(a, int)),
    "float?": Def(1, lambda a: isinstance(a, float)),
    "and": Def(2, None),
    "quote": Def(1, None),
    "or": Def(2, None),
    "if": Def(3, None),
    "not": Def(1, operator.not_),
    "err": Def(1, err),
    "tl": Def(1, lambda a: a[1:]),
    "/": Def(2, operator.truediv),
    "rnd-float": Def(0, lambda: random.random()),
    "rnd-int": Def(1, lambda n: random.randrange(n)),
    "rnd-choice": Def(1, lambda s: random.choice(s)),
    "pr": Def(1, pr),
    "prn": Def(1, prn),
}


class Break(Exception):
    pass


def ev(a, env):
    if isinstance(a, str):
        if a in env:
            return env[a]

        r = defs[a].val
        if r is None:
            raise Exception(a)
        return r
    if isinstance(a, tuple):
        o = a[0]

        if o == "=":
            val = ev(a[2], env)
            env[a[1]] = val
            return val
        if o == "do":
            return evs(a[1:], env)
        if o == "break":
            raise Break()
        if o == "loop":
            for i in range(1000):
                try:
                    evs(a[1:], env)
                except Break:
                    break
            return env["result"]
        if o == "and":
            return ev(a[1], env) and ev(a[2], env)
        if o == "\\":
            params = a[1]
            body = a[2]

            def f(*args):
                e = env.copy()
                for key, val in zip(params, args):
                    e[key] = val
                return ev(body, e)

            return f
        if o == "fn":
            name = a[1]
            if name in env:
                raise Exception(name)
            params = a[2]
            body = ("do",) + a[3:]

            def f(*args):
                e = env.copy()
                for key, val in zip(params, args):
                    e[key] = val
                return ev(body, e)

            env[name] = f
            return
        if o == "if":
            return ev(a[2], env) if ev(a[1], env) else ev(a[3], env)
        if o == "or":
            return ev(a[1], env) or ev(a[2], env)
        if o == "quote":
            return a[1]

        f = ev(o, env)
        args = [ev(b, env) for b in a[1:]]
        return f(*args)
    return a


def evs(s, env):
    r = 0
    for a in s:
        r = ev(a, env)
    return r


env = {}
for key in defs:
    d = defs[key]
    if d.val is not None:
        env[key] = d.val

filename = "etc.k"
evs(parse(), env)

filename = "test.k"
evs(parse(), env)

print("ok")
