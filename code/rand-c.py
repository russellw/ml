import argparse
import os
import random
import subprocess
import tempfile

# command line
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random number seed", type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

# files
c_file = os.path.join(tempfile.gettempdir(), "a.c")
asm_file = os.path.join(tempfile.gettempdir(), "a.asm")

# loop
epoch = 0
while 1:
    if epoch % 10000 == 0:
        print(epoch)

    n = random.randint(1, 10)
    v = []
    for i in range(n):
        while 1:
            c = chr(random.randrange(127))
            if c.isprintable() or c == "\n":
                break
        v.append(c)
    s = "".join(v)
    open(c_file, "w").write(s)

    cmd = "cl", "/Fa", "/c", c_file
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
    stdout, stderr = p.communicate()
    if not stderr and not p.returncode:
        print(repr(s))

    epoch += 1
