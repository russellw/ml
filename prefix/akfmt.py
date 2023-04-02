import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="*")
args = parser.parse_args()
# tokenizer
def constituent(c):
    if c.isspace():
        return
    if c in "()[];{}":
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
        v = []
        while not eat(")"):
            if not tok:
                raise Exception("unclosed '('")
            v.append(expr())
        return tuple(v)
    s = tok
    lex()
    return s


def do(filename):
    global toks
    global text
    global ti
    toks = []
    ti = 0
    text = open(filename).read()
    lex()

    v = []
    while tok:
        v.append(expr())
    print(v)


for f in args.files:
    for root, dirs, files in os.walk(f):
        for filename in files:
            if os.path.splitext(filename)[1] == ".k":
                do(os.path.join(root, filename))
