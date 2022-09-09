import argparse
import os
import subprocess


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
        raise Exception(str(p.returncode))
    return stdout


def compare(prg1, prg2):
    s = call(prg1)
    t = call(prg2)
    if s != t:
        print(s, end="")
        print(t, end="")
        exit(1)


def exe_file(f):
    if os.name == "nt":
        return f + ".exe"
    return f


parser = argparse.ArgumentParser(description="Run separate compilation test")
parser.add_argument("--clang", help="path to clang")
args = parser.parse_args()

clang = r"C:\llvm-project\build\Debug\bin\clang"
if args.clang:
    clang = args.clang

test_dir = os.path.dirname(os.path.realpath(__file__))

v = []
for root, dirs, files in os.walk(test_dir):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext == ".c":
            subprocess.check_call(
                [
                    clang,
                    "-S",
                    "-emit-llvm",
                    os.path.join(root, f),
                ]
            )
            f = os.path.splitext(f)[0]
            v.append(f + ".ll")
subprocess.check_call([clang, "-o", exe_file("a0")] + v)
subprocess.check_call(["olivine"] + v)
subprocess.check_call([clang, "a.ll"])
compare("a0", "a")
print("ok")
