import argparse
import tempfile
import re
import os
import subprocess
import hashlib

parser = argparse.ArgumentParser(description="solve for link libraries")
parser.add_argument("-d", action="store_true", help="debug build")
parser.add_argument("obj", help="the object file")
parser.add_argument("dirs", nargs="+", help="dirs to look for libs")
args = parser.parse_args()


def add(d, k, x):
    if k not in d:
        d[k] = []
    d[k].append(x)


def hashFile(f):
    with open(f, "rb") as f:
        h = hashlib.sha512()
        blocksize = 1 << 20
        while 1:
            b = f.read(blocksize)
            if not b:
                return h.hexdigest()
            h.update(b)


def dumpbin(opts, f):
    if isinstance(opts, str):
        opts = [opts]
    cmd = ["dumpbin"] + opts + [f]
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


def isx64(f):
    s = dumpbin("/headers", f)

    # the explicit marker is the most reliable way to tell
    if "8664 machine (x64)" in s:
        return 1

    # sometimes actual x64 libraries show up as unknown machine
    if "0 machine (Unknown)" in s:
        # but then again, so do non-x64 ones
        # so fall back on guessing by directory name
        if "x64" in f:
            return 1


# find libs that are x64 and not duplicate
hs = {}
print("finding libs")
for di in args.dirs:
    for root, dirs, files in os.walk(di):
        if "$Recycle.Bin" in root:
            continue
        for f in files:
            if not f[-4:].lower() == ".lib":
                continue
            f = os.path.join(root, f)
            print(f, end="\t")
            if not isx64(f):
                print("not x64")
                continue
            h = hashFile(f)
            add(hs, h, f)
            if len(hs[h]) > 1:
                print("duplicate")
                continue
            print("ok")
ndups = 0
libs = []
for h in hs.keys():
    v = hs[h]
    ndups += len(v) - 1
    v.sort()
    libs.append(v[0])
    print(v[0])
    for s in v[1:]:
        print(s + "\tduplicate")
print(str(len(libs)) + " libs")
print(str(ndups) + " duplicates")


# Find symbols in libraries
def getSymbols(f):
    v = dumpbin("/linkermember:1", f).splitlines()
    r = set()
    i = 0
    while 1:
        s = v[i]
        i += 1
        m = re.match(r"\s*(\d+) public symbols", s)
        if m:
            n = int(m[1])
            break
    i += 1
    while i < len(v):
        s = v[i]
        i += 1
        m = re.match(r"\s*[0-9A-F]+ (.*)", s)
        if not m:
            break
        r.add(m[1].rstrip())
    return r


def getExports(f):
    v = dumpbin("/exports", f).splitlines()
    r = set()
    i = 0
    while 1:
        if i == len(v):
            return r
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
        r.add(s)
    return r


symbols = {}
exports = {}
print()
print("finding symbols")
for f in libs:
    print(f, end="\t")

    es = getExports(f)
    print(len(es), end="\t")
    for s in es:
        add(exports, s, f)

    ss = getSymbols(f)
    print(len(ss))
    for s in ss:
        if s in es:
            continue
        add(symbols, s, f)
print(str(len(symbols)) + " symbols")
print(str(len(exports)) + " exports")


# Start with the libraries the compiler wants to use by default
# https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
print()
f = os.path.join(tempfile.gettempdir(), "empty-program.c")
with open(f, "w") as g:
    g.write("main() {}\n")
cmd = "cl /c "
if args.d:
    cmd += "/MTd "
subprocess.check_call(cmd + f)
p = subprocess.Popen(
    "link /verbose empty-program.obj",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
stdout, stderr = p.communicate()
stdout = str(stdout, "utf-8")
stderr = str(stderr, "utf-8")
if p.returncode:
    raise Exception(str(p.returncode))
libs = []
for s in stdout.splitlines():
    m = re.match(r"    Searching (.*\.lib):", s, re.IGNORECASE)
    if not m:
        continue
    s = m[1]
    if s not in libs:
        libs.append(s)
print("default libs", end="\t")
print(libs)


# try linking with a given list of libraries
def link(libs):
    cmd = ["link", "/nodefaultlib", args.obj, "@objs.lst"] + libs
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = str(stdout, "utf-8")
    stderr = str(stderr, "utf-8")
    if p.returncode and p.returncode != 1120:
        raise Exception(str(p.returncode))
    print(stdout, end="")
    r = []
    for s in stdout.splitlines():
        # https://stackoverflow.com/questions/68814812/what-is-the-difference-between-lnk2001-and-lnk2019-errors-in-visual-studio
        m = re.match(r".* : error LNK2001: unresolved external symbol (.*)", s)
        if m:
            r.append(m[1].rstrip())
            continue
        m = re.match(
            r".* : error LNK2019: unresolved external symbol (.*) referenced in function ",
            s,
        )
        if m:
            r.append(m[1])
            continue
    return r


# Preference order when several libraries offer a symbol
def libkey(f):
    st = os.stat(f)
    return os.path.basename(f), -st.st_mtime, f


# Iteratively choose a symbol, then a library that offers it
while 1:
    print()
    print("linking", end="\t")
    print(libs)
    v = link(libs)
    print("missing", end="\t")
    print(set(v))
    if not v:
        break
    s = v[0]
    print("choosing", end="\t")
    print(s)
    if s in symbols:
        fs = symbols[s]
    elif s in exports:
        fs = exports[s]
    else:
        print("not found anywhere")
        exit(1)
    fs.sort(key=libkey)
    print("in", end="\t")
    print(fs)
    f = fs[0]
    libs.append(f)

# write result
print()
print("writing libs.lst")
with open("libs.lst", "w") as f:
    for s in libs:
        s = '"' + s + '"'
        f.write(s + "\n")
        print(s)
