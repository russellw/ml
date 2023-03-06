import os
import sys

out = open("a", "wb")
for root, dirs, fs in os.walk(sys.argv[1]):
    for f in fs:
        if os.path.splitext(f)[1] in (".c", ".h"):
            file = os.path.join(root, f)
            s = open(file, "rb").read()
            out.write(s)
