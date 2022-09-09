import argparse
import sys
import os
import subprocess


parser = argparse.ArgumentParser(
    description="edit olivine.h to include only needed LLVM headers"
)
parser.add_argument("-i", action="store_true", help="inplace edit")
args = parser.parse_args()

if not args.i:
    print("-i not specified, taking no action")
    exit(1)


def writeLines(filename, lines):
    with open(filename, "w", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")


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


# Read existing meta-header
s = os.path.dirname(sys.argv[0])
assert s.endswith("etc")
src = s[:-3] + "src"
mh = os.path.join(src, "olivine.h")
lines = open(mh).read().splitlines()

# cut existing header list
i = 0
while not lines[i].startswith("#include <llvm/"):
    i += 1
hi = i
while lines[i].startswith("#include <llvm/"):
    i += 1
headers = lines[hi:i]
del lines[hi:i]

# Paste candidate new header list
def paste(hs):
    lines1 = lines[:]
    lines1[hi:hi] = hs
    return lines1


# Does a candidate reduced header list still let the program compile?
flags = call("llvm-config --cxxflags").rstrip()


def ok(hs):
    writeLines(mh, paste(hs))
    cmd = "cl /c " + os.path.join(src, "*.cpp") + " " + flags
    r = subprocess.run(cmd)
    return r.returncode == 0


# check whether each header is needed
i = 0
while i < len(headers):
    hs = headers[:]
    del hs[i]
    if ok(hs):
        headers = hs
    else:
        i += 1

# write new meta-header
writeLines(mh, paste(headers))
