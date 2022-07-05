import random

symbols = ("+", "-", "*", "//", "0", "1")


def rand(size):
    code = []
    for i in range(size):
        a = random.choice(symbols)
        code.append(a)
    return code


def run(code):
    stack = []

    def pop():
        if stack:
            return stack.pop()
        return 0

    for a in code:
        if a.isdigit():
            stack.append(int(a))
            continue
        y = pop()
        x = pop()
        if a == "//" and y == 0:
            z = 0
        else:
            z = eval(str(x) + a + str(y))
        stack.append(z)
    return pop()


if __name__ == "__main__":
    for i in range(10):
        code = rand(10)
        print(code)
        print(run(code))
