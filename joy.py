import random

symbols = ("+", "-", "*", "//", "0", "1", "(", ")", "if", "lambda", "a0", "a1")


def rand(size):
    code = []
    for i in range(size):
        a = random.choice(symbols)
        code.append(a)
    return code


def parse(code):
    i = 0

    def expr():
        nonlocal i
        a = code[i]
        i += 1
        if a != "(":
            return a
        v = []
        while a[i] != ")":
            v.append(expr())
        i += 1
        return v

    return expr()


if __name__ == "__main__":
    for i in range(20):
        code = rand(10)
        print(code)
        a = parse(code)
