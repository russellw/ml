__all__ = [
    "parse",
]

# tokenizer
text = ""
ti = 0
tok = None


def constituent(c):
    if c.isalnum():
        return 1
    return c in "_+-*/?"


def lex():
    global ti
    global tok
    while ti < len(text):
        i = ti

        # whitespace
        if text[ti].isspace():
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
        raise Exception("stray '%c' in program" % text[ti])
    tok = None


# parser
def eat(k):
    if tok == k:
        lex()
        return 1


def expr():
    if tok == "(":
        v = []
        lex()
        while not eat(")"):
            v.append(expr())
        return tuple(v)
    if tok[0].isdigit():
        a = int(tok)
        lex()
        return a
    s = tok
    lex()
    return s


def parse(filename):
    global text
    global ti
    text = open(filename).read() + "\n"
    ti = 0
    lex()
    v = []
    while tok:
        v.append(expr())
    return v
