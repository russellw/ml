import argparse
import sys
import os
import time
import subprocess

parser = argparse.ArgumentParser(description="run benchmarks")
parser.add_argument(
    "interpreters", nargs="+", help="list of interpreters whose speed is to be compared"
)
args = parser.parse_args()


def call(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = str(stdout, "utf-8")
    if p.returncode:
        stderr = str(stderr, "utf-8")
        print(stdout, end="")
        print(stderr, end="")
        raise Exception(str(p.returncode))
    return stdout


def bench(it, b):
    start = time.time()
    call((it, b))
    return time.time() - start


def geomean(xs):
    r = 1.0
    for x in xs:
        r *= x
    return r ** (1.0 / len(xs))


tss = []
for it in args.interpreters:
    print("\t" + it, end="")
    tss.append([])
print()

bd = os.path.join(os.path.dirname(os.path.realpath(__file__)), "benchmarks")
for root, dirs, files in os.walk(bd):
    for f in files:
        if not f[-4:].lower() == ".lua":
            continue
        print(f, end="")
        sys.stdout.flush()
        f = os.path.join(root, f)
        ts = []
        for it in args.interpreters:
            t = bench(it, f)
            print("\t" + str(t), end="")
            ts.append(t)
        print()
        for i in range(len(ts)):
            tss[i].append(ts[i])

for i in range(len(tss)):
    print("\t" + str(geomean(tss[i])), end="")
print()
