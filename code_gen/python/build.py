import argparse
import os
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-d", action="store_true", help="debug build")
parser.add_argument(
    "--incremental",
    action="store_true",
    help="only recompile modules where source file is newer than IR",
)
parser.add_argument("src")
args = parser.parse_args()


src = os.path.realpath(args.src)


def flattenPath(f):
    s = f
    if s[0].isalpha() and s[1] == ":":
        s = s[2:]
    s = s.replace("\\", "_")
    s = s.replace("/", "_")
    if s.startswith("_"):
        s = s[1:]
    return s


def unquote(s):
    if s.startswith('"'):
        assert s.endswith('"')
        s = s[1:-1]
    v = []
    i = 0
    while i < len(s):
        if s[i : i + 2] == '\\"':
            i += 2
            v.append('"')
            continue
        if s[i : i + 2] == "\\\\":
            i += 2
            v.append("\\")
            continue
        c = s[i]
        i += 1
        if c == '"':
            continue
        v.append(c)
    return "".join(v)


# object files that come from assembly files
# these need to be used as they are
objs = []


# C source files to LLVM compiled modules
# these need to be recompiled with clang
modules = []


def cc(opts, f):
    if not f[-2:].lower() == ".c":
        print("skipping " + f)
        return
    if os.path.basename(f) in ("launcher.c",):
        print("skipping " + f)
        return

    outf = flattenPath(f)
    outf = os.path.splitext(outf)[0] + ".ll"
    modules.append(outf)

    try:
        if args.incremental and os.path.getmtime(f) < os.path.getmtime(outf):
            return
    except FileNotFoundError:
        pass

    cmd = ["clang"] + opts
    if os.path.isabs(f):
        cmd.append(f)
    else:
        cmd.append(os.path.join(src, f))
    cmd.append("-O3")
    cmd.append("-S")
    cmd.append("-emit-llvm")
    cmd.append("-w")
    cmd.append("-o")
    cmd.append(outf)
    subprocess.check_call(cmd)


def cl(s):
    # assuming paths don't contain spaces
    v = s.split()

    # try to unwrap arguments in quotes
    # though this is not the end of the matter
    # because sometimes quotes are also used within an option
    v = [unquote(s) for s in v]

    # compiler command line will include options for things like include directories
    # which we need to copy
    opts = []

    # compiler command line may specify multiple source files
    # which we need to gather in order to process individually
    files = []

    i = 0
    while i < len(v):
        s = v[i]
        i += 1

        # depending on how the project is set up, '..' might not occur in the build at all
        # but if it does, the best guess is that it refers to the project root directory
        s = s.replace("..", src)

        # option
        if s.startswith("/"):
            s = s[1:]

            # sometimes /D has a space after it
            if s == "D":
                opts.append("-" + s)
                opts.append(v[i])
                i += 1
                continue

            # and sometimes it doesn't
            if s[0] in ("D", "I"):
                # but sometimes it does have a quote after it
                opts.append("-" + s[0] + unquote(s[1:]))
                continue
            continue

        # not an option, so presumably a source file
        files.append(s)

    for f in files:
        cc(opts, f)


# link steps to the record of which dynamic libraries contain which modules
dlls = []


def link(dll, ext):
    global modules
    dll = unquote(dll)
    dll = os.path.basename(dll)
    dll = os.path.splitext(dll)[0] + ext
    for s in modules:
        ds = dll, s
        if ds in dlls:
            raise Exception(str(ds))
        dlls.append(ds)
    modules = []


# scan msbuild.log
lines = open(os.path.join(src, "msbuild.log")).readlines()
for s in lines:
    m = re.match(r".*\btracker\.exe", s, re.IGNORECASE)
    if m:
        continue

    m = re.match(r'.*\bml64 .* /Fo (".*\.obj") ".*\.asm"', s, re.IGNORECASE)
    if m:
        objs.append(m[1])
        continue

    m = re.match(r".*\bcl\.exe", s, re.IGNORECASE)
    if m:
        cl(s[len(m[0]) :])
        continue

    m = re.match(r".*\blink\.exe.*/IMPLIB:(\S+).*/DLL", s)
    if m:
        link(m[1], ".dll")
        continue

    m = re.match(r".*\blink\.exe.*/IMPLIB:(\S+)", s)
    if m:
        link(m[1], ".exe")
        continue

# write lists
with open("objs.lst", "w") as f:
    for s in objs:
        f.write(s + "\n")
with open("modules.tsv", "w") as f:
    for dll, s in dlls:
        f.write(dll + "\t" + s + "\n")

# next steps
main = "python.exe"
if args.d:
    main = "python_d.exe"
subprocess.check_call(f"olivine -link-only -main={main} modules.tsv")
subprocess.check_call("clang -c a.ll")
