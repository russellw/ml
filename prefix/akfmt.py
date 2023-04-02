import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="*")
args = parser.parse_args()

# tokenizer
def constituent(c):
    if c.isspace():
        return
    if c in "()[]{};":
        return
    return 1


def lex():
    global ti
    global tok
    while ti < len(text):
        # whitespace
        if text[ti].isspace():
            ti += 1
            continue

        i = ti

        # comment
        if text[ti] == ";":
            while text[ti] != "\n":
                ti += 1
        elif text[ti] == "{":
            while text[ti] != "}":
                ti += 1
            ti += 1

        # word
        elif constituent(text[ti]):
            while constituent(text[ti]):
                ti += 1

        # punctuation
        else:
            ti += 1

        tok = text[i:ti]
        return
    tok = None


# parser
def eat(k):
    if tok == k:
        lex()
        return 1


def expr():
    if eat("("):
        a = []
        while not eat(")"):
            if not tok:
                raise Exception("unclosed '('")
            a.append(expr())
        return a
    a = tok
    lex()
    return a


# printer
def list_depth(a):
    if isinstance(a, list):
        return max(map(list_depth, a), default=0) + 1
    return 0


def any_rec(f, a):
    if isinstance(a, list):
        for b in a:
            if any_rec(f, b):
                return 1
        return
    return f(a)


def indent(n):
    out.append("  " * n)


def want_vertical(a):
    if isinstance(a, list) and a:
        if any_rec(lambda b: b.startswith(";"), a):
            return 1
        if list_depth(a) > 5:
            return 1


def horizontal(a):
    if isinstance(a, list):
        out.append("(")
        if a:
            out.append(a[0])
            for b in a[1:]:
                out.append(" ")
                horizontal(b)
        out.append(")")
        return
    out.append(a)


def vertical(a, dent):
    assert isinstance(a, list)
    out.append("(")
    dent += 1
    for b in a[1:]:
        out.append("\n")
        indent(dent)
        pprint1(b, dent)
    out.append(")")


def pprint1(a, dent):
    if want_vertical(a):
        vertical(a, dent)
    else:
        horizontal(a)


def pprint(a):
    pprint1(a, 0)
    out.append("\n")


# top level
def do(filename):
    global toks
    global text
    global ti
    global out

    # read
    toks = []
    ti = 0
    text = open(filename).read()
    lex()

    # parse
    v = []
    while tok:
        v.append(expr())

    # print
    out = []
    for a in v:
        pprint(a)
    out = "".join(out)
    print(out, end="")


for f in args.files:
    if os.path.isdir(f):
        for root, dirs, files in os.walk(f):
            for filename in files:
                if os.path.splitext(filename)[1] == ".k":
                    do(os.path.join(root, filename))
        continue
    do(f)
