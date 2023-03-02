import argparse
import os
import subprocess
import tempfile

exts = set()
exts.add(".exe")


# command line
parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# files
def do(file):
    out = os.path.join(tempfile.gettempdir(), "a.asm")
    cmd = "dumpbin", "/disasm", file, "/nologo", "/out:" + out
    subprocess.check_call(cmd)
    print("%s\t%d\t%d" % (file, os.stat(file).st_size, os.stat(out).st_size))


for s in args.files:
    if os.path.isdir(s):
        for root, dirs, files in os.walk(s):
            for file in files:
                if os.path.splitext(file)[1] in exts:
                    do(os.path.join(root, file))
        continue
    do(s)
