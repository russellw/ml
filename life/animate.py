import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from life import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--density", help="density of random grid", type=float, default=0.5
)
parser.add_argument("-r", "--rand", help="random pattern", action="store_true")
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("-x", help="origin X coordinate", type=int)
parser.add_argument("-y", help="origin Y coordinate", type=int)
parser.add_argument("-z", "--size", help="grid size", type=int, default=400)
parser.add_argument("file", nargs="?")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

size = args.size

origin = -(size // 2)
if args.rand:
    origin = 0
x0 = args.x
y0 = args.y
if x0 is None:
    x0 = origin
if y0 is None:
    y0 = origin

x1 = x0 + size
y1 = y0 + size

a = None
if args.rand is not None:
    a = rand(size, args.density)
if args.file:
    a = read(args.file)


def update(frame):
    global a
    a = run(a)
    img.set_data(matrix(a, x0, y0, x1, y1))
    return img


fig, ax = plt.subplots()
img = ax.imshow(matrix(a, x0, y0, x1, y1), interpolation="nearest")
ani = animation.FuncAnimation(fig, update, interval=0)
plt.show()
