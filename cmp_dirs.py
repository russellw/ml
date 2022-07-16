import argparse
import os
import hashlib

parser = argparse.ArgumentParser(description="compare two directories")
parser.add_argument("dir1")
parser.add_argument("dir2")
args = parser.parse_args()


def hashFile(path):
    with open(path, "rb") as f:
        h = hashlib.sha512()
        blocksize = 1 << 20
        while 1:
            b = f.read(blocksize)
            if not b:
                return h.hexdigest()
            h.update(b)


def subpath(d, f):
    i = len(d) + 1
    return f[i:]


def getfiles(d):
    v = []
    for root, dirs, files in os.walk(d):
        for f in files:
            f = os.path.realpath(os.path.join(root, f))
            f = subpath(d, f)
            v.append(f)
    return v


d1 = os.path.realpath(args.dir1)
d2 = os.path.realpath(args.dir2)

v1 = set(getfiles(d1))
v2 = set(getfiles(d2))
v = sorted(list(v1 | v2))

w = 0
for s in v:
    w = max(w, len(s))


def printStat(f):
    try:
        st = os.stat(f)
        print("%9d" % st.st_size, end="")
    except FileNotFoundError:
        print(9 * " ", end="")


for f in v:
    f1 = os.path.join(d1, f)
    f2 = os.path.join(d2, f)
    try:
        if hashFile(f1) == hashFile(f2):
            continue
    except FileNotFoundError:
        pass
    printStat(f1)
    print(" ", end="")
    print(f.rjust(w), end="")
    print(" ", end="")
    printStat(f2)
    print()
