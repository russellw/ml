import argparse
import os
import re
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


def cc(f):
    call(
        (
            "cl",
            "/DDEBUG",
            "/EHsc",
            "/I" + os.path.join(here, "lib"),
            "/W3",
            "/WX",
            "/Zi",
            "/nologo",
            f,
            os.path.join(here, "lib", "*.cc"),
            "dbghelp.lib",
        ),
        20,
    )


def do(f):
    print(f)
    cc(f)
    s = call("test")
    print(s)

    f = os.path.join(tempfile.gettempdir(), "a.cc")
    open(f, "wb").write(s)
    cc(f)
    subprocess.check_call("a")


for root, dirs, files in os.walk(here):
    for f in files:
        if f == "test.cc":
            do(os.path.join(root, f))
