import os
import math
import matplotlib.pyplot as plt

problems = []
for root, dirs, files in os.walk("tsplib/"):
    for filename in files:
        if os.path.splitext(filename)[1] != ".tsp":
            continue
        filename = root + filename
        lines = open(filename).readlines()
        for s in lines:
            if s.startswith("EDGE_WEIGHT_TYPE"):
                t = s.split(":")[1].strip()
                if t == "EUC_2D":
                    problems.append(filename)
                    break


def parse_num(s):
    return int(float(s))


def parse(filename):
    lines = open(filename).readlines()
    lines = [s.strip() for s in lines]
    i = lines.index("NODE_COORD_SECTION")
    j = len(lines)
    while lines[j - 1] in ("", "EOF"):
        j -= 1
    p = lines[i + 1 : j]
    assert p
    p = [s.split() for s in p]
    p = [(parse_num(c[0]), parse_num(c[1])) for c in p]
    assert len(p) == len(set(p))
    return p


def dist(c, d):
    return math.sqrt((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2)


def length(p):
    t = 0.0
    for i in range(len(p) - 1):
        t += dist(p[i], p[i + 1])
    return t


for filename in problems[:3]:
    p = parse(filename)
    xs = [c[0] for c in p]
    ys = [c[1] for c in p]
    plt.plot(xs, ys, "bo")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
