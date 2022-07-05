import random

symbols = ("+", "-", "*", "/", "0", "1")


def rand():
    v = []
    for i in range(10):
        a = random.choice(symbols)
        v.append(a)
    return v


def run(code):
    stack = []
    for a in code:
        if a.isdigit():
            stack.append(int(a))
        elif len(stack) < 2:
            stack = [0]
        else:
            x = stack[-2]
            y = stack[-1]
            if a == "/" and y == 0:
                z = 0
            else:
                z = eval(str(x) + a + str(y))
            stack = stack[:-2] + [z]
    return stack[-1]


if __name__ == "__main__":
    for i in range(10):
        code = rand()
        print(code)
        print(run(code))
