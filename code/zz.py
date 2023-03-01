import os
import random

import torch

alphabet_size = 126 - 31 + 1


def chop(v, size):
    r = []
    for i in range(0, len(v) - (size - 1), size):
        r.append(v[i : i + size])
    return r


def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    sys.stderr.write(f"{info.filename}:{info.function}:{info.lineno}: {a}\n")


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

    c -= 31
    if c < alphabet_size:
        v.append(c)


def encodes(s):
    if isinstance(s, str):
        s = s.encode()
    v = []
    for c in s:
        encode1(v, c)
    return v


def get_filenames(exts, s):
    r = []
    if os.path.splitext(s)[1] in exts:
        r.append(s)
    for root, dirs, files in os.walk(s):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                r.append(os.path.join(root, file))
    return r


def one_hot(a):
    v = [0.0] * alphabet_size
    v[a] = 1.0
    return v


def print_dl(dl):
    for x, y in dl:
        print("x:")
        print(x)
        print(x.shape)
        print(x.dtype)
        print()

        print("y:")
        print(y)
        print(y.shape)
        print(y.dtype)
        break


def read_chunks(file, size):
    return chop(read_file(file), size)


def read_file(file):
    s = open(file, "rb").read()
    return encodes(s)


def scramble(v, n):
    v = v.copy()
    for i in range(n):
        j = random.randrange(len(v))
        k = random.randrange(len(v))
        v[j], v[k] = v[k], v[j]
    return v


def split_train_test(v):
    i = len(v) * 80 // 100
    return v[:i], v[i:]


def tensor(v):
    r = []
    for a in v:
        r.extend(one_hot(a))
    return torch.as_tensor(r)


# unit tests
assert len(encodes("\r")) == 0
assert len(encodes("\n")) == 1
assert encodes("\t") == encodes(" ")
assert encodes("\t") != encodes("a")
assert encodes("~")[0] == alphabet_size - 1

assert chop("abcd", 2) == ["ab", "cd"]
assert chop("abcd", 3) == ["abc"]
