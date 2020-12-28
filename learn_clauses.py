import datetime
import logging
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


def debug(a):
    logger.debug(str(a), stack_info=True)


pr(sys.argv)
pr()


class Var:
    pass


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
        if tok != o:
            raise ValueError(tok)
        lex()

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

    def args():
        expect("(")
        r = []
        if tok != ")":
            r.append(atomic_term())
            while tok == ",":
                lex()
                r.append(atomic_term())
        expect(")")
        return tuple(r)

    def atomic_term():
        o = tok

        # defined
        if o[0] == "$":
            lex()
            return o

        # variable
        if o[0].isupper():
            lex()
            return Var()

        # function
        a = read_name()
        if tok == "(":
            s = args()
            return (a,) + s
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
pr(len(neg))
pr(len(pos))

pr(f"{time.time() - start:.3f} seconds")
