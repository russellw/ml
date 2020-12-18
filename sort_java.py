#!/usr/bin/python3
# Sort elements of Java code
# Assumes indent already run
# Does not work for all possible Java programs
# Test carefully before reusing in other projects

import argparse
import os
import re

parser = argparse.ArgumentParser(description="Sort elements of Java code")
parser.add_argument("files", nargs="+")
args = parser.parse_args()


def read_lines(filename):
    with open(filename) as f:
        return [s.rstrip("\n") for s in f]


def write_lines(filename, lines):
    with open(filename, "w") as f:
        for s in lines:
            f.write(s + "\n")


def flatten(xss):
    r = []
    for xs in xss:
        for x in xs:
            r.append(x)
    return r


def terminated(i, indent):
    if re.match(indent + "}", lines[i]):
        return True
    if re.match(indent + "}", lines[i - 1]):
        i -= 1
    while re.match(r"\s*$", lines[i - 1]) or re.match(r"\s*//", lines[i - 1]):
        i -= 1
    if re.match(r"\s*System.exit\(", lines[i - 1]):
        return True
    if re.match(r"\s*break;", lines[i - 1]):
        return True
    if re.match(r"\s*continue;", lines[i - 1]):
        return True
    if re.match(r"\s*return( |;)", lines[i - 1]):
        return True
    if re.match(r"\s*throw ", lines[i - 1]):
        return True


def case(i, indent):
    while True:
        while re.match(r"\s*$", lines[i]) or re.match(r"\s*//", lines[i]):
            i += 1
        if not re.match(indent + "(case .*|default):", lines[i]):
            raise ValueError(filename + ":" + str(i) + ": case not found")
        while re.match(indent + "(case .*|default):$", lines[i]):
            i += 1
        if re.match(indent + "(case .*|default): {", lines[i]):
            while not re.match(indent + "}", lines[i]):
                i += 1
            i += 1
        else:
            while not re.match(indent + "(case .*|default)", lines[i]) and not re.match(
                indent + "}", lines[i]
            ):
                i += 1
        if terminated(i, indent):
            return i


def cases(i, indent):
    r = []
    while not re.match(indent + "}", lines[i]):
        j = case(i, indent)
        r.append(lines[i:j])
        i = j
    return i, r


def sort_case(c):
    i = 0
    while re.match(r"\s*(case .*|default):$", c[i]):
        i += 1
    brace = re.match(r"\s*(case .*|default): {", c[i])
    if brace:
        c[i] = c[i][:-2]
        i += 1
    c[:i] = sorted(c[:i])
    if brace:
        c[i - 1] += " {"


def sort_switch1(i, indent):
    j, cs = cases(i, indent)
    for c in cs:
        sort_case(c)
    cs = sorted(cs)
    lines[i:j] = flatten(cs)


def sort_switch():
    for i in range(len(lines)):
        m = re.match(r"(\s*)switch (.*) {", lines[i])
        if not m:
            continue
        indent = m[1]
        sort_switch1(i + 1, indent)


def sort_file():
    global lines
    lines = read_lines(filename)
    old = lines[:]
    sort_switch()
    if lines == old:
        return
    print(filename)
    write_lines(filename, lines)


for arg in args.files:
    if os.path.isfile(arg):
        filename = arg
        sort_file()
        continue
    for root, dirs, files in os.walk(arg):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext != ".java":
                continue
            filename = os.path.join(root, filename)
            sort_file()
