#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser(description="calculate density of DIMACS files")
parser.add_argument("list_file")
args = parser.parse_args()


def read_lines(filename):
    with open(filename) as f:
        return [s.rstrip("\n") for s in f]


def read_dimacs(filename):
    global variables
    global clauses
    variables = set()
    clauses = []
    for s in read_lines(filename):
        if not s:
            continue
        if s[0].isalpha():
            continue
        s = [int(t) for t in s.split()]
        c = []
        for a in s:
            if not a:
                break
            variables.add(abs(a))
            c.append(a)
        clauses.append(c)
    variables = sorted(list(variables))


files = read_lines(args.list_file)
for filename in files:
    read_dimacs(filename)
    width = variables[-1]
    full = 0
    for c in clauses:
        full += len(c)
    space = width * len(clauses)
    density = full / space
    print(
        f"{len(variables):10,d}, {len(clauses):10,d}: {full:20,d} / {space:20,d} = {density:g}"
    )
