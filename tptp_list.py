import sys
import os
import random

s = []
for root, dirs, files in os.walk("TPTP"):
    for fname in files:
        filename = os.path.join(root, fname)
        if os.path.splitext(filename)[1] != ".p":
            continue
        if "^" in filename:
            continue
        s.append(filename)
random.shuffle(s)
for t in s[: int(sys.argv[1])]:
    print(t)
