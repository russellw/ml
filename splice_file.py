import sys

i = int(sys.argv[1])
xs = open(sys.argv[2]).readlines()
ys = open(sys.argv[3]).readlines()
xs[i:i] = ys
open(sys.argv[4], "w").writelines(xs)
