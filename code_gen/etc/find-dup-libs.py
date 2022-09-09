import argparse
import os
import hashlib
from datetime import datetime

parser = argparse.ArgumentParser(description="find duplicate libs")
parser.add_argument("dirs", nargs="+", help="dirs to look for libs")
args = parser.parse_args()


def add(d, k, x):
    if k not in d:
        d[k] = []
    d[k].append(x)


def hashFile(path):
    with open(path, "rb") as f:
        h = hashlib.sha512()
        blocksize = 1 << 20
        while 1:
            b = f.read(blocksize)
            if not b:
                return h.hexdigest()
            h.update(b)


dups = {}

for di in args.dirs:
    for root, dirs, files in os.walk(di):
        for f in files:
            if not f[-4:].lower() == ".lib":
                continue
            f = os.path.realpath(os.path.join(root, f))
            h = hashFile(f)
            add(dups, h, f)
for h in sorted(dups.keys(), key=lambda h: sorted(dups[h])):
    v = sorted(dups[h])
    if len(v) < 2:
        continue
    st = os.stat(v[0])
    print(st.st_size)
    for f in v:
        st = os.stat(f)
        t = datetime.fromtimestamp(st.st_mtime)
        print(str(t) + "\t" + f)
    print()
