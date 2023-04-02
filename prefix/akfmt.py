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
def is_comment(a):
    return isinstance(a, str) and a[0] in ";{"


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
    out.append("\n")
    out.append("  " * n)


def want_vertical(a):
    # TODO rec
    if isinstance(a, list) and a:
        if any_rec(is_comment, a):
            return 1
        if a[0] in ("fn",):
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


header_len = {
    "\\": 1,
    "fn": 2,
    "if": 1,
    "when": 1,
}


def blank_between(a, b):
    if not is_comment(a) and (is_comment(b) or b[0] == "fn"):
        return 1


def verticals(a, dent):
    for i in range(len(a)):
        if i:
            if blank_between(a[i - 1], a[i]):
                out.append("\n")
            indent(dent)
        pprint1(a[i], dent)


def vertical(a, dent):
    assert isinstance(a, list)
    out.append("(")

    n = 1 + header_len.get(a[0], 0)
    horizontal(a[0])
    for b in a[1:n]:
        out.append(" ")
        horizontal(b)

    dent += 1
    indent(dent)
    verticals(a[n:], dent)

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
    global out
    global text
    global ti
    global toks

    # read
    toks = []
    ti = 0
    text = open(filename).read()
    lex()

    # parse
    a = []
    while tok:
        a.append(expr())

    # print
    out = []
    verticals(a, 0)
    out.append("\n")
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
