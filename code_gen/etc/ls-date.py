import sys
import time
import os

d = "."
if len(sys.argv) > 1:
    d = sys.argv[1]

fs = []
for root, dirs, files in os.walk(d):
    if root.startswith(".\\."):
        continue
    for f in files:
        f = os.path.join(root, f)
        if f.startswith(".\\"):
            f = f[2:]
        t = os.path.getmtime(f)
        n = os.path.getsize(f)
        fs.append((t, n, f))
fs.sort()

for t, n, f in fs:
    print(str(time.ctime(t)) + ("%10d" % n) + "  " + f)
