# minimal implementation of DPLL with only unit propagation
# baseline for comparison
import argparse
import os
import re
import time


def read_lines(filename):
    with open(filename) as f:
        return [s.rstrip("\n") for s in f]


# read


def read_dimacs(filename):
    global clauses
    global expected_sat
    global variables

    # header
    expected_nvariables = -1
    expected_nclauses = -1
    expected_sat = None

    # data
    variables = set()
    clauses = []
    c = set()
    for s in read_lines(filename):
        if not s:
            continue
        if s[0] == "c":
            if "UNSAT" in s:
                expected_sat = False
            elif "SAT" in s:
                expected_sat = True
            continue
        if s[0] == "p":
            m = re.match(r"p\s+cnf\s+(\d+)\s+(\d+)", s)
            expected_nvariables = int(m[1])
            expected_nclauses = int(m[2])
            continue
        if s[0] != "-" and not s[0].isdigit():
            raise ValueError(s)
        for k in s.split():
            k = int(k)
            if k == 0:
                clauses.append(list(c))
                c.clear()
                continue
            a = k, True
            if k < 0:
                k = -k
                a = k, False
            variables.add(k)
            c.add(a)
    if c:
        clauses.append(list(c))
    if expected_nclauses >= 0 and len(clauses) != expected_nclauses:
        raise ValueError(str(len(clauses)))
    variables = sorted(variables)


# search


class Node:
    def __init__(self, parent, k, v):
        self.parent = parent
        assign(self, k, v)


def assign(node, k, v):
    node.k = k
    node.v = v
    node.units = []
    A = assigned(node)
    while True:
        a = unit(A)
        if not a:
            break
        node.units.append(a)
        k, v = a
        A[k] = v


def assigned(node):
    A = {}
    while node:
        assert node.k not in A
        A[node.k] = node.v
        for k, v in node.units:
            assert k not in A
            A[k] = v
        node = node.parent
    return A


def conflict(A):
    for c in clauses:
        if conflictc(A, c):
            return c


def conflictc(A, c):
    for k, v in c:
        if k not in A or A[k] is v:
            return False
    return True


def true(A, c):
    for k, v in c:
        if A.get(k) is v:
            return True


def unassigned(A):
    for k in variables:
        if k not in A:
            return k


def unit(A):
    for c in clauses:
        if true(A, c):
            continue
        d = []
        for k, v in c:
            if k not in A:
                d.append((k, v))
        if len(d) == 1:
            return d[0]


# solve


def solve():
    node = None
    i = 0
    while True:
        i += 1
        A = assigned(node)

        # conflict?
        if conflict(A):
            while node and node.v:
                node = node.parent
            if not node:
                print(str(i) + " iterations")
                return None
            node = Node(node.parent, node.k, True)
            continue

        # satisfied?
        if len(A) == len(variables):
            print(str(i) + " iterations")
            return A

        # choose a variable
        k = unassigned(A)
        assert k

        # assign it
        # and propagate units
        node = Node(node, k, False)


# main


def do(filename):
    if os.path.splitext(filename)[1] == ".lst":
        for s in read_lines(filename):
            do(s)
        return
    print(filename)
    read_dimacs(filename)
    print(f"{len(variables)} variables")
    print(f"{len(clauses)} clauses")
    start = time.process_time()
    A = solve()
    if A is None:
        print("unsat")
    else:
        print("sat")
        for k in sorted(A.keys()):
            if not A[k]:
                print("-", end="")
            print(k, end="")
            print(" ", end="")
        print()
        assert not conflict(A)
    if isinstance(expected_sat, bool) and isinstance(A, dict) != expected_sat:
        raise ValueError(A)
    t = time.process_time() - start
    print(f"{t:.3f} seconds")
    print()


parser = argparse.ArgumentParser(description="SAT solver")
parser.add_argument("files", nargs="+")
args = parser.parse_args()
for arg in args.files:
    if os.path.isfile(arg):
        do(arg)
        continue
    for root, dirs, files in os.walk(arg):
        for filename in files:
            do(os.path.join(root, filename))
