import inspect


def isConcrete(a):
    if isinstance(a, int) or isinstance(a, str):
        return 1
    if not isinstance(a, tuple):
        return
    if len(a) >= 1000000:
        return
    return all(map(isConcrete, a))


def compose(a):
    s = []

    def rec(a):
        if not isinstance(a, tuple):
            s.append(a)
            return
        s.append("(")
        for b in a:
            rec(b)
        s.append(")")

    rec(a)
    return s


def size(a):
    if not isinstance(a, tuple):
        return 1
    return sum(map(size, a))


def deBruijn(a, env=("x0",)):
    # local variable
    if a in env:
        return "arg", env.index(a)

    # atom
    if not isinstance(a, tuple) or not a:
        return a

    # lambda
    o = a[0]
    if o == "lambda":
        params = a[1]
        a = deBruijn(a[2], tuple(reversed(params)) + env)
        for x in params:
            a = o, a
        return a

    # function call or other special form
    return tuple(deBruijn(b, env) for b in a)


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def bitLen(n):
    assert n >= 0
    bits = 0
    while n:
        n >>= 1
        bits += 1
    return bits


def composeBits(a, vocab, bits=0):
    # symbol designating an integer
    int1 = len(vocab)

    # number of bits per word
    if not bits:
        bits = bitLen(int1)

    # output numbers are limited to the range that fits in a word
    unsignedBits = bits - 1
    max1 = (1 << unsignedBits) - 1

    # twos complement
    min1 = -max1 - 1

    # output bit stream
    s = []

    # write one word as bits
    def write(n):
        t = []
        for i in range(bits):
            t.append(n & 1)
            n >>= 1

        # big endian for easier debugging
        s.extend(reversed(t))

    # tokenize and write words
    for a in compose(a):
        if isinstance(a, str):
            write(vocab.index(a))
            continue
        if isinstance(a, int):
            write(int1)
            a = max(a, min1)
            a = min(a, max1)
            write(a)
            continue
        raise ValueError(a)

    assert len(s) % bits == 0
    return s


if __name__ == "__main__":
    assert isConcrete(1)
    assert isConcrete(True)
    assert isConcrete("a")
    assert isConcrete((1, 2, 3))
    assert not isConcrete((1, 2, len))

    assert size(5) == 1
    assert size("abc") == 1
    assert size(("abc", "def")) == 2

    assert compose(3) == [3]
    assert compose(("+", 3, "x0")) == ["(", "+", 3, "x0", ")"]

    assert deBruijn(("+", 3, "x0")) == ("+", 3, ("arg", 0))
    assert deBruijn(("lambda", ("x1", "x2"), ("+", "x0", ("*", "x1", "x2")))) == (
        "lambda",
        ("lambda", ("+", ("arg", 2), ("*", ("arg", 1), ("arg", 0)))),
    )

    assert bitLen(1) == 1
    assert bitLen(2) == 2
    assert bitLen(3) == 2

    assert composeBits(3, (), 4) == [0, 0, 0, 0, 0, 0, 1, 1]
    assert composeBits(7, (), 4) == [0, 0, 0, 0, 0, 1, 1, 1]
    assert composeBits(8, (), 4) == [0, 0, 0, 0, 0, 1, 1, 1]
    assert composeBits(-1, (), 4) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert composeBits(-8, (), 4) == [0, 0, 0, 0, 1, 0, 0, 0]
    assert composeBits(-9, (), 4) == [0, 0, 0, 0, 1, 0, 0, 0]
    assert composeBits(3, ("a", "b"), 4) == [0, 0, 1, 0, 0, 0, 1, 1]

    print("ok")
