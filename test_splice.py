def size(a):
    if isinstance(a, list) or isinstance(a, tuple):
        n = 0
        for b in a:
            n += size(b)
        return n
    return 1


def splice(a, i, b):
    if not i:
        return b
    assert isinstance(a, list) or isinstance(a, tuple)
    r = []
    for x in a:
        n = size(x)
        if 0 <= i < n:
            x = splice(x, i, b)
        i -= n
        r.append(x)
    return tuple(r)


assert splice("a", 0, "b") == "b"
assert splice(("+", "x", "y"), 0, "b") == "b"
assert splice(("+", "x", "y"), 1, "b") == ("+", "b", "y")
assert splice(("+", "x", "y"), 2, "b") == ("+", "x", "b")
print("ok")
