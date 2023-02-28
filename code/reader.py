import os


def get_filenames(exts, src_dir):
    v = []
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if os.path.splitext(filename)[1] in exts:
                v.append(os.path.join(root, filename))
    return v


def encode1(v, c):
    # CR LF = LF
    if c == 10:
        v.append(0)
        return
    if c == 13:
        return

    # tab = space
    if c == 9:
        v.append(1)
        return

    # printable char
    assert c >= 32
    v.append(c - 31)


def encodes(s):
    if isinstance(s, str):
        s = s.encode()
    v = []
    for c in s:
        encode1(v, c)
    return v


def get_file(filename):
    s = open(filename, "rb").read()
    return encodes(s)


def chop(v, size):
    r = []
    for i in range(0, len(v) - (size - 1), size):
        r.append(v[i : i + size])
    return r


# unit tests
assert len(encodes("\r")) == 0
assert len(encodes("\n")) == 1
assert encodes("\t") == encodes(" ")
assert encodes("\t") != encodes("a")

assert chop("abcd", 2) == ["ab", "cd"]
assert chop("abcd", 3) == ["abc"]
