import sys


def read_file(filename):
    clauses = []
    for s in open(filename).readlines():
        if not s:
            continue
        if s[0].isalpha():
            continue
        neg = []
        pos = []
        for a in s.split():
            if a == "0":
                continue
            if a[0] == "-":
                neg.append(a[1:])
            else:
                pos.append(a)
        if neg or pos:
            neg = tuple(neg)
            pos = tuple(pos)
            clauses.append((neg, pos))
    return tuple(clauses)


if __name__ == "__main__":
    clauses = read_file(sys.argv[1])
    print(clauses)
