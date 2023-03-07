import sys

for c in open(sys.argv[1], "rb").read():
    if c > 126:
        print(c)
        exit(1)
