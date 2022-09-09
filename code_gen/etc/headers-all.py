import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
    description="edit olivine.h to include all LLVM headers"
)
parser.add_argument("-i", action="store_true", help="inplace edit")
args = parser.parse_args()


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


# list of LLVM include directories
s = call("llvm-config --includedir").rstrip()
assert s.endswith("llvm/include") or s.endswith("llvm\\include")
hdirs = [s, s[:-12] + "build/include"]

# list of LLVM headers
headers = []
for hdir in hdirs:
    for root, dirs, files in os.walk(hdir):
        for filename in files:
            if os.path.splitext(filename)[1] == ".h":
                # exclude any that are known to conflict with other headers
                if filename in ["ItaniumDemangle.h"]:
                    continue
                r = root[len(hdir) + 1 :].replace("\\", "/")
                filename = r + "/" + filename
                headers.append(filename)
headers.sort()

# keep only headers that compile without special requirements
flags = call("llvm-config --cxxflags").rstrip()


def ok(hs):
    with open("a.cc", "w") as f:
        for s in hs:
            f.write("#include <" + s + ">\n")
    cmd = "cl /c a.cc " + flags
    r = subprocess.run(cmd)
    return r.returncode == 0


hs = []
for s in headers:
    if ok(hs + [s]):
        hs.append(s)
headers = hs

# no in-place edit, just print results
if not args.i:
    for s in headers:
        print(s)
    exit(0)

# Read existing meta-header
main_h = os.path.join(args.src, "olivine.h")
lines = open(main_h).read().splitlines()

# delete existing header list
i = 0
while not lines[i].startswith("#include <llvm/"):
    i += 1
hi = i
while lines[i].startswith("#include <llvm/"):
    i += 1
del lines[hi:i]

# paste new header list
lines[hi:hi] = ["#include <" + s + ">" for s in headers]

# write new meta-header
writeLines(main_h, lines)
