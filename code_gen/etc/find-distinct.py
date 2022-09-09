import sys
import hashlib


def hashFile(path):
    with open(path, "rb") as f:
        h = hashlib.sha512()
        blocksize = 1 << 20
        while 1:
            b = f.read(blocksize)
            if not b:
                return h.hexdigest()
            h.update(b)


files = {}

for s in sys.stdin:
    s = s.split("\t")[0]
    h = hashFile(s)
    if h not in files:
        files[h] = s
for s in sorted(files.values()):
    print(s)
