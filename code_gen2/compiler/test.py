import glob
import os
import subprocess
import tempfile


def call(cmd, limit=0):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = str(stdout, "utf-8").replace("\r\n", "\n")
    stderr = str(stderr, "utf-8").replace("\r\n", "\n")
    if stderr:
        raise Exception(stderr)
    if p.returncode:
        if limit:
            stdout = "\n".join(stdout.split("\n")[:limit])
        print(stdout, end="")
        raise Exception(str(p.returncode))
    return stdout


here = os.path.dirname(os.path.realpath(__file__))
lib = os.path.join(here, "..", "lib")
exe = os.path.join(tempfile.gettempdir(), "a")

tests = []
for root, dirs, files in os.walk(os.path.join(here, "test")):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext == ".cpp":
            f = os.path.join(root, f)
            tests.append(f)


def cco(f):
    if os.name == "nt":
        cmd = [
            "cl",
            "/EHsc",
            "/Fo" + exe,
            "/I" + lib,
            "/W4",
            "/WX",
            "/c",
            "/nologo",
            f,
        ]
    else:
        cmd = [
            "g++",
            "-I" + lib,
            "-Wall",
            "-Werror",
            "-Wextra",
            "-c",
            "-o" + exe,
        ]
        cmd.extend(list(glob.glob(f)))
    call(cmd, 20)


# Compile all test files with the C++ compiler (without linking)
# to make sure they are valid C++
for f in tests:
    cco(f)


def cc(f):
    if os.name == "nt":
        cmd = [
            "cl",
            "/DDEBUG",
            "/EHsc",
            "/Fe" + exe,
            "/I" + lib,
            "/W4",
            "/WX",
            "/Zi",
            "/nologo",
            f,
            os.path.join(lib, "*.cc"),
            "dbghelp.lib",
        ]
    else:
        cmd = [
            "g++",
            "-DDEBUG",
            "-I" + lib,
            "-Wall",
            "-Werror",
            "-Wextra",
            "-o" + exe,
        ]
        cmd.extend(list(glob.glob(f)))
        cmd.extend(list(glob.glob(os.path.join(lib, "*.cc"))))
    call(cmd, 20)


# Compile the Olivine compiler
f = os.path.join(here, "*.cc")
cc(f)

# Smoke test
s = call((exe, "-h"))
assert s

# Run the Olivine compiler on all test files
for f in tests:
    s = call((exe, f))
    print(s)
exit(0)

s = call((exe, here))

f = os.path.join(tempfile.gettempdir(), "a.cc")
open(f, "w", newline="\n").write(s)
cc(f)
subprocess.check_call(exe)
