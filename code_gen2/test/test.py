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
    if stderr:
        stderr = str(stderr, "utf-8")
        raise Exception(stderr)
    if p.returncode:
        stdout = str(stdout, "utf-8").replace("\r\n", "\n")
        if limit:
            stdout = "\n".join(stdout.split("\n")[:limit])
        print(stdout)
        raise Exception(str(p.returncode))
    return stdout


here = os.path.dirname(os.path.realpath(__file__))
lib = os.path.join(here, "..", "lib")


def cc(f):
    call(
        (
            "cl",
            "/DDEBUG",
            "/EHsc",
            "/I" + lib,
            "/W3",
            "/WX",
            "/Zi",
            "/nologo",
            f,
            os.path.join(lib, "*.cc"),
            "dbghelp.lib",
        ),
        20,
    )


f = os.path.join(here, "test.cc")
cc(f)
s = call("test")

f = os.path.join(tempfile.gettempdir(), "a.cc")
open(f, "wb").write(s)
cc(f)
subprocess.check_call("a")
