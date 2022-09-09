import argparse

parser = argparse.ArgumentParser(
    description="check a TSV file for duplicate values of one field"
)
parser.add_argument("--col", default="B", help="column, default=B")
parser.add_argument("filename", help="the TSV file")
args = parser.parse_args()

if args.col.isupper():
    j = ord(args.col) - ord("A")
elif args.col.islower():
    j = ord(args.col) - ord("a")
else:
    j = int(args.col) - 1

d = {}

for s in open(args.filename).readlines():
    u = s.split("\t")
    k = u[j]
    if k not in d:
        d[k] = []
    d[k].append(s)
for k, v in d.items():
    if len(v) > 1:
        for s in v:
            print(s, end="")
