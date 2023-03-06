import os
import subprocess
import tempfile

file = os.path.join(tempfile.gettempdir(), "a.c")
# file = os.path.join(tempfile.gettempdir(), "a.java")
# file = os.path.join(tempfile.gettempdir(), "a.go")

# cmd = "cl", "/c", file
# cmd = "javac", file
# cmd = "go", file
cmd = "tcc", "-c", file

open(file, "w").write("")
for i in range(100):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()

# cl: 45 ms
# javac: 700 ms
# go: 83 ms
# tcc: 18 ms
