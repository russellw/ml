import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(
    description="find libs, dumpbin /exports to exports.tsv"
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
    v = call('dumpbin /exports "' + f + '"').splitlines()
    i = 0
    while 1:
        if i == len(v):
            return
        s = v[i]
        i += 1
        if "Exports" in s:
            break
    while 1:
        s = v[i]
        i += 1
        if "ordinal    name" in s:
            break
    while not v[i].rstrip():
        i += 1
    while v[i].rstrip():
        s = v[i].rstrip()
        i += 1
        s = s[18:]
        syms.append((f, s))


if os.path.isdir(args.path):
    for root, dirs, files in os.walk(args.path):
        for f in files:
            if not f[-4:].lower() == ".lib":
                continue
            f = os.path.join(root, f)
            do(f)
else:
    do(args.path)

with open("exports.tsv", "w") as f:
    for w in syms:
        f.write(w[0] + "\t" + w[1] + "\n")
