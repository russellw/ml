import argparse

parser = argparse.ArgumentParser(
    description="find how many lines two files have in common"
)
parser.add_argument("file1")
parser.add_argument("file2")
args = parser.parse_args()


def norm(s):
    return s.replace("/", "\\")


u = open(args.file1).read().splitlines()
v = open(args.file2).read().splitlines()
i = 0
while i < len(u) and i < len(v):
    if norm(u[i]) != norm(v[i]):
        print(u[i])
        print(v[i])
        print(i)
        exit(0)
    i += 1
