import argparse
import re

parser = argparse.ArgumentParser(
    description="normalize printed representations of pointers in debug output"
)
parser.add_argument("-i", action="store_true", help="inplace edit")
parser.add_argument("filename")
args = parser.parse_args()

d = {"0" * 16: "#"}


def norm(s):
    if s not in d:
        d[s] = "#%d" % len(d)
    return d[s]


r = re.compile("[0-9A-F]" * 16)
v = []
for s in open(args.filename).read().splitlines():
    w = []
    i = 0
    while i < len(s):
        if r.match(s[i : i + 16]):
            w.append(norm(s[i : i + 16]))
            i += 16
            continue
        w.append(s[i])
        i += 1
    v.append("".join(w))

if args.i:
    with open(args.filename, "w") as f:
        for s in v:
            f.write(s + "\n")
else:
    for s in v:
        print(s)
