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
        print(stdout)
        raise Exception(str(p.returncode))
    return stdout


here = os.path.dirname(os.path.realpath(__file__))
lib = os.path.join(here, "..", "lib")
exe = os.path.join(tempfile.gettempdir(), "a")


def cc(f):
    if os.name == "nt":
        cmd = [
            "cl",
            "/DDEBUG",
            "/EHsc",
            "/Fe" + exe,
            "/I" + lib,
            "/W3",
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
            f,
        ] + list(glob.glob(os.path.join(lib, "*.cc")))
    call(cmd, 20)


f = os.path.join(here, "test.cc")
cc(f)
s = call(exe)

f = os.path.join(tempfile.gettempdir(), "a.cc")
open(f, "w", newline="\n").write(s)
cc(f)
subprocess.check_call(exe)
