import argparse
import os

parser = argparse.ArgumentParser(description="compare two directories for added files")
parser.add_argument("dir1")
parser.add_argument("dir2")
args = parser.parse_args()


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

for f in sorted(list(v2)):
    if f in v1:
        continue
    f = os.path.join(d2, f)
    print(f)
