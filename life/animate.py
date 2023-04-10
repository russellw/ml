import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from life import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--density", help="density of random grid", type=float, default=0.5
)
parser.add_argument("-g", "--steps", help="number of steps", type=int, default=1000)
parser.add_argument("-r", "--rand", help="random pattern size", type=int)
parser.add_argument("-s", "--seed", help="random number seed", type=int)
parser.add_argument("file", nargs="?")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

g = None
if args.rand is not None:
    g = randgrid(args.rand, args.density)
if args.file:
    g = read(args.file)

size=100

def update(frame):
    g.run()
    data=g.get_data()
    img = ax.imshow(data, interpolation="nearest")
    img.set_data(data)
    return (img,)

fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig,
    update,
    interval=100,
)
plt.show()
