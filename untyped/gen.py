import random

from interpreter import defs, run


def simplify(a):
    return a


def mk(depth):
    if depth == 0 or random.random() < 0.10:
        return random.choice((0, 1, (), "x"))
    o = random.choice(list(defs))
    v = [o]
    for i in range(defs[o].arity):
        v.append(mk(depth - 1))
    return simplify(tuple(v))


seen = set()


def print_new(a):
    if a not in seen:
        print(a)
        seen.add(a)


if __name__ == "__main__":
    random.seed(0)

    x = 10, 20, 30
    for i in range(1000):
        a = mk(5)
        print_new(a)
        try:
            if run(a, {"x": x}) == 30:
                print("***", i)
                break
        except (IndexError, ZeroDivisionError):
            pass

    print("ok")
