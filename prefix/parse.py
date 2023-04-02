__all__ = [
    "parse",
]

# tokenizer
filename = ""
text = ""
ti = 0
line = 0
tok = None


def constituent(c):
    if c.isalnum():
        return 1
    return c in "_+-*/?=<>"


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


def parse(filename1):
    global text
    global filename
    global line
    global ti
    filename = filename1
    text = open(filename).read() + "\n"
    ti = 0
    line = 1
    lex()
    v = []
    while tok:
        v.append(expr())
    return v
