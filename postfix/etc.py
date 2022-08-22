import inspect


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


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {repr(a)}")


def fixLen(s, n):
    return s[:n] + [0] * (n - len(s))


def bitLen(n):
    assert n >= 0
    bits = 0
    while n:
        n >>= 1
        bits += 1
    return bits


def toBits(s, vocab, bits=0):
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
    r = []

    # write one word as bits
    def write(n):
        t = []
        for i in range(bits):
            t.append(n & 1)
            n >>= 1

        # big endian for easier debugging
        r.extend(reversed(t))

    # tokenize and write words
    for a in s:
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

    assert len(r) % bits == 0
    return r


if __name__ == "__main__":
    assert compose(3) == [3]
    assert compose(("+", 3, "x0")) == ["(", "+", 3, "x0", ")"]

    assert bitLen(1) == 1
    assert bitLen(2) == 2
    assert bitLen(3) == 2

    assert toBits(3, (), 4) == [0, 0, 0, 0, 0, 0, 1, 1]
    assert toBits(7, (), 4) == [0, 0, 0, 0, 0, 1, 1, 1]
    assert toBits(8, (), 4) == [0, 0, 0, 0, 0, 1, 1, 1]
    assert toBits(-1, (), 4) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert toBits(-8, (), 4) == [0, 0, 0, 0, 1, 0, 0, 0]
    assert toBits(-9, (), 4) == [0, 0, 0, 0, 1, 0, 0, 0]
    assert toBits(3, ("a", "b"), 4) == [0, 0, 1, 0, 0, 0, 1, 1]

    assert fixLen([1, 2, 3], 5) == [1, 2, 3, 0, 0]
    assert fixLen([1, 2, 3], 2) == [1, 2]

    print("ok")
