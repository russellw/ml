import datetime
import inspect
import logging
import random
import sys
import time

start = time.time()

logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler(datetime.datetime.now().strftime("logs/%Y-%m-%d %H%M%S.log"))
)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def pr(a=""):
    logger.info(str(a))


def db(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    logger.debug(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


pr(sys.argv)
pr()


class Var:
    pass


variable = Var()


def fns(a):
    s = set()

    def get_fn(a):
        if isinstance(a, Fn):
            s.add(a)

    walk(get_fn, a)
    return s


def walk(f, a):
    if isinstance(a, tuple):
        for b in a:
            walk(f, b)
    f(a)


def parse(text):
    # tokenizer
    ti = 0
    tok = ""

    def lex():
        nonlocal ti
        nonlocal tok
        while ti < len(text):
            c = text[ti]

            # space
            if c.isspace():
                ti += 1
                continue

            # word
            if c.isalpha() or c == "$":
                i = ti
                ti += 1
                while text[ti].isalnum() or text[ti] == "_":
                    ti += 1
                tok = text[i:ti]
                return

            # quote
            if c == "'":
                i = ti
                ti += 1
                while text[ti] != c:
                    if text[ti] == "\\":
                        ti += 1
                    ti += 1
                ti += 1
                tok = text[i:ti]
                return

            # punctuation
            if text[ti : ti + 2] in ("!=",):
                tok = text[ti : ti + 2]
                ti += 2
                return
            tok = c
            ti += 1
            return

        # end of file
        tok = None

    def eat(o):
        if tok == o:
            lex()
            return True

    def expect(o):
        if not eat(o):
            raise ValueError(tok)

    # terms
    def read_name():
        o = tok

        # word
        if o[0].islower():
            lex()
            return o

        # single quoted, equivalent to word
        if o[0] == "'":
            lex()
            return o[1:-1]

        raise ValueError(o)

    def atomic_term():
        o = tok

        # defined
        if o[0] == "$":
            lex()
            return o

        # variable
        if o[0].isupper():
            lex()
            return variable

        # function/constant
        a = read_name()
        if eat("("):
            r = [a, atomic_term()]
            while eat(","):
                r.append(atomic_term())
            expect(")")
            return tuple(r)
        return a

    def infix_unary():
        a = atomic_term()
        o = tok
        if o == "=":
            lex()
            return "=", a, atomic_term()
        if o == "!=":
            lex()
            return "~", ("=", a, atomic_term())
        return a

    def unitary_formula():
        o = tok
        if o == "~":
            lex()
            return "~", unitary_formula()
        return infix_unary()

    # clause
    lex()
    expect("cnf")
    expect("(")
    lex()
    expect(",")
    lex()
    expect(",")
    expect("(")
    r = [unitary_formula()]
    while eat("|"):
        r.append(unitary_formula())
    return tuple(r)


neg = []
pos = []
for s in open(sys.argv[1]):
    if not s.startswith("cnf"):
        continue
    a = parse(s)
    if s.endswith("trainneg\n"):
        neg.append(a)
    elif s.endswith("trainpos\n"):
        pos.append(a)
    else:
        continue
n = min(len(neg), len(pos))
neg = neg[:n]
pos = pos[:n]
pr(len(neg))
pr(len(pos))
pr()


def term_size(a):
    if isinstance(a, tuple):
        n = 0
        for b in a:
            n += term_size(b)
        return n
    return 1


def clause_size(c):
    n = len(c) * 1000
    for a in c:
        n += term_size(a)
    return n


pr(sum(map(clause_size, neg)))
pr(sum(map(clause_size, pos)))
pr()

cs = [(c, clause_size(c), 0) for c in neg] + [(c, clause_size(c), 1) for c in pos]
random.shuffle(cs)
cs.sort(key=lambda a: a[1])


def acc(i):
    n = 0
    for j in range(i):
        if cs[j][2] == 0:
            n += 1
    for j in range(i, len(cs)):
        if cs[j][2] == 1:
            n += 1
    return n


for i in range(0, len(cs), 100):
    pr(f"{i}\t{acc(i)}")
pr()

pr(f"{time.time() - start:.3f} seconds")
