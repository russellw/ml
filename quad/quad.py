import inspect
import math
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
def constituent(c):
    if c.isspace():
        return
    if c in "()[];{}":
        return
    return 1


def lex():
    global line
    global ti
    global tok
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
            while text[ti] != "}":
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


def parse_expr():
    lin = line
    if eat("("):
        a = []
        while not eat(")"):
            a.append(parse_expr())
        a = tuple(a)
        if not a:
            return a
        if a[0] == "assert":
            return (
                "if",
                a[1],
                0,
                ("err", ("quote", ("%s:%d: assert failed" % (filename, lin)))),
            )
        return a
    if tok[0].isdigit() or (tok[0] == "-" and len(tok) > 1 and tok[1].isdigit()):
        a = int(tok)
        lex()
        return a
    a = tok
    lex()
    return a


def parse():
    global line
    global text
    global ti
    text = open(filename).read() + "\n"
    ti = 0
    line = 1
    lex()
    a = []
    while tok:
        a.append(parse_expr())
    return a


# compiler
def assoc(a):
    if len(a) <= 3:
        return a
    o = a[0]
    return (o, a[1], assoc(((o,) + a[2:])))


def pairwise(a):
    if len(a) == 3:
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


def comp_expr(a, code):
    if isinstance(a, int):
        return "quote", a
    if isinstance(a, tuple):
        # expand syntax sugar
        if a[0] in ("+", "*", "and", "or"):
            a = assoc(a)
        elif a[0] in ("==", "/=", "<", "<=", ">", ">="):
            a = pairwise(a)

        if a[0] == "and":
            r = gensym()
            a = (
                "do",
                ("=", r, a[1]),
                ("if", r, a[2], r),
            )
        elif a[0] == "or":
            r = gensym()
            a = (
                "do",
                ("=", r, a[1]),
                ("if", r, r, a[2]),
            )
        elif a[0] == "when":
            a = "if", a[1], (("do",) + a[1:]), 0

        # special form
        if a[0] == "if":
            r = gensym()

            # condition
            cond = comp_expr(a[1], code)
            fixup_false = len(code)
            code.append(["if-not", cond])

            # true
            code.append(["=", r, comp_expr(a[2], code)])
            fixup_after = len(code)
            code.append(["goto"])

            # false
            code[fixup_false].append(len(code))
            code.append(["=", r, comp_expr(a[3], code)])

            # after
            code[fixup_after].append(len(code))
            return r
        elif a[0] == "quote":
            return a

        # function args
        v = [a[0]]
        for b in a[1:]:
            v.append(comp_expr(b, code))

        # special form
        if a[0] == "do":
            return v[-1]

        # function call
        a = len(code)
        code.append(v)
    return a


def err(s):
    raise Exception(s)


class Fn:
    def __init__(self, prim=None):
        self.prim = prim


fns = {
    "*": Fn(operator.mul),
    "+": Fn(operator.add),
    "-": Fn(operator.sub),
    "/": Fn(operator.truediv),
    "<": Fn(operator.lt),
    "<=": Fn(operator.le),
    "==": Fn(operator.eq),
    "at": Fn(lambda a, b: a[int(b)]),
    "cons": Fn(lambda a, b: (a,) + b),
    "div": Fn(operator.floordiv),
    "err": Fn(err),
    "sqrt": Fn(math.sqrt),
    "float?": Fn(lambda a: isinstance(a, float)),
    "hd": Fn(lambda a: a[0]),
    "int?": Fn(lambda a: isinstance(a, int)),
    "len": Fn(lambda a: len(a)),
    "list?": Fn(lambda a: isinstance(a, tuple)),
    "mod": Fn(operator.mod),
    "neg": Fn(operator.neg),
    "not": Fn(operator.not_),
    "pow": Fn(operator.pow),
    "choice": Fn(random.choice),
    "random": Fn(random.random),
    "randrange": Fn(lambda n: random.randrange(int(n))),
    "sym?": Fn(lambda a: isinstance(a, str)),
    "tl": Fn(lambda a: a[1:]),
}


def comp(name, params, body):
    fn = Fn()
    fn.params = params
    fn.code = []
    r = comp_expr(("do",) + body, fn.code)
    fn.code.append(("ret", r))
    fns[name] = fn
    return fn


# interpreter
def get(env, a):
    if isinstance(a, tuple):
        assert a[0] == "quote"
        return a[1]
    return env[a]


def call(fn, args):
    if fn.prim:
        return fn.prim(*args)

    assert len(fn.params) == len(args)
    env = {}
    for key, val in zip(fn.params, args):
        env[key] = val
    i = 0
    while 1:
        a = fn.code[i]
        if a[0] == "=":
            env[a[1]] = get(env, a[2])
            i += 1
            continue
        if a[0] == "goto":
            i = a[1]
            continue
        if a[0] == "if":
            i = a[2] if get(env, a[1]) else i + 1
            continue
        if a[0] == "if-not":
            i = i + 1 if get(env, a[1]) else a[2]
            continue
        if a[0] == "ret":
            return get(env, a[1])
        r = call(fns[a[0]], [get(env, b) for b in a[1:]])
        env[i] = r
        i += 1


def ev(a, env):
    if isinstance(a, tuple):
        o = a[0]

        if o == "loop":
            for i in range(1000):
                try:
                    evs(a[1:], env)
                except Break:
                    break
            return env["result"]
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

        f = ev(o, env)
        args = [ev(b, env) for b in a[1:]]
        return f(*args)
    return a


main = []
for filename in ("etc.k", "test.k"):
    for a in parse():
        if a[0] == "fn":
            comp(a[1], a[2], a[3:])
            continue
        main.append(a)
main = tuple(main)
main = comp("main", (), main)
call(main, ())
print("ok")
