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


def want_vertical(a):
    if isinstance(a, list):
        if not a:
            return
        if a[0] in ("\\", "do", "fn", "loop", "when"):
            return 1
        if any(map(want_vertical, a)):
            return 1
        if list_depth(a) > 5:
            return 1
        return
    return is_comment(a)


header_len = {
    "\\": 1,
    "fn": 2,
    "if": 1,
    "when": 1,
}


def blank_between(a, b):
    if a[0] == "fn":
        return 1
    if not is_comment(a) and (is_comment(b) or b[0] == "fn"):
        return 1


def indent(n):
    out.append("\n")
    out.append("  " * n)


def vertical(a, dent):
    for i in range(len(a)):
        if i:
            if blank_between(a[i - 1], a[i]):
                out.append("\n")
            indent(dent)
        pprint(a[i], dent)


def pprint(a, dent=0):
    if isinstance(a, list):
        out.append("(")
        if a:
            n = len(a)
            if want_vertical(a):
                n = 1 + header_len.get(a[0], 0)

            pprint(a[0])
            for b in a[1:n]:
                out.append(" ")
                pprint(b)

            if n < len(a):
                dent += 1
                indent(dent)
                vertical(a[n:], dent)
        out.append(")")
        return
    out.append(a)


# top level
def do(filename):
    global out
    global text
    global ti
    global toks

    # read
    text = open(filename).read()
    ti = 0
    toks = []

    # parse
    lex()
    a = []
    while tok:
        a.append(expr())

    # print
    out = []
    vertical(a, 0)
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
