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
        if text[ti].isdigit():
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
        if a[0] == "assert":
            a = (
                "if",
                a[1],
                0,
                ("err", ("quote", ("%s:%d: assert failed" % (filename, line1)))),
            )
        return a
    if tok[0].isdigit():
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
