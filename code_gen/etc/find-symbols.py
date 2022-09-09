import argparse
import re
import os
import subprocess
import sys

parser = argparse.ArgumentParser(
    description="find libs, dumpbin /symbols to symbols.tsv"
)
parser.add_argument("path")
args = parser.parse_args()


def call(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = str(stdout, "utf-8")
    stderr = str(stderr, "utf-8")
    if stderr:
        raise Exception(stderr)
    if p.returncode:
        # dumpbin uses this return code to indicate that the program has been successfully run
        # but it does not recognize the file as a valid library file
        if p.returncode == 1136:
            return ""
        raise Exception(str(p.returncode))
    return stdout


syms = []


def do(f):
    f = os.path.realpath(f)
    print(f)
    v = call('dumpbin /symbols "' + f + '"').splitlines()
    for s in v:
        if not re.match(r"\w\w\w\s*\w\w\w\w\w\w\w\w\s*", s):
            continue
        w = s.split("|")
        if len(w) != 2:
            continue
        r = []
        r.append(f)
        name = w[1].strip()
        w = w[0].split()
        r.append(w[2])
        r.append(w[3])
        r.append(w[4])
        r.append(name)
        syms.append(r)


if os.path.isdir(args.path):
    for root, dirs, files in os.walk(args.path):
        for f in files:
            if not f[-4:].lower() == ".lib":
                continue
            f = os.path.join(root, f)
            do(f)
else:
    do(args.path)

with open("symbols.tsv", "w") as f:
    for w in syms:
        f.write("\t".join(w) + "\n")
