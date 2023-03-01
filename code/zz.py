import os
import random

import torch

alphabet_size = 126 - 31 + 1


def get_chunks(exts, src_dir, size):
    r = []
    for filename in get_filenames(exts, src_dir):
        r.extend(chop(get_file(filename), size))
    return r


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


def get_file(filename):
    s = open(filename, "rb").read()
    return encodes(s)


def chop(v, size):
    r = []
    for i in range(0, len(v) - (size - 1), size):
        r.append(v[i : i + size])
    return r


def one_hot(a):
    v = [0.0] * alphabet_size
    v[a] = 1.0
    return v


def tensor(v):
    r = []
    for a in v:
        r.extend(one_hot(a))
    return torch.as_tensor(r)


def scramble(v, n):
    v = v.copy()
    for i in range(n):
        j = random.randrange(len(v))
        k = random.randrange(len(v))
        v[j], v[k] = v[k], v[j]
    return v


# unit tests
assert len(encodes("\r")) == 0
assert len(encodes("\n")) == 1
assert encodes("\t") == encodes(" ")
assert encodes("\t") != encodes("a")
assert encodes("~")[0] == alphabet_size - 1

assert chop("abcd", 2) == ["ab", "cd"]
assert chop("abcd", 3) == ["abc"]
