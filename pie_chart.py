import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt


# args
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="CSV file")
args = parser.parse_args()


# data
y_train = []
with open(args.filename, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for r in reader:
        y_train.append(r[-1])

# output
ys = set()
for y in y_train:
    ys.add(y)
ys = sorted(list(ys))
out_features = len(ys)
print(f"y: {len(y_train)}; {out_features} possible values")

y_frequency = {}
for y in ys:
    y_frequency[y] = 0
for y in y_train:
    y_frequency[y] += 1
print(y_frequency)

ys.sort(key=lambda a: y_frequency[a])

y_frequency = {}
for y in ys:
    y_frequency[y] = 0
for y in y_train:
    y_frequency[y] += 1
print(y_frequency)

# adapted from
# https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = y_frequency.keys()
sizes = y_frequency.values()
explode = [0] * len(ys)
# explode [-1]= 0.1

fig1, ax1 = plt.subplots()
ax1.pie(
    sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
)
ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig(os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".png")
plt.show(block=False)
plt.pause(10)
