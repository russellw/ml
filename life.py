import random
import torch

def randboard():
    size=10
    b=[]
    for i in range(size):
        b.append([random.randrange(2) for j in range(size)])
    return b

def get(b,i,j):
    size=len(b)
    if i<0 or i>=size:
        return 0
    if j<0 or j>=size:
        return 0
    return b[i][j]

def step(b):
    size=len(b)
    b1=[]
    for i in range(size):
        row=[]
        for    j  in range(size):
            n=0
            for i1 in range(i-1,i+2):
                for j1 in range(j-1,j+2):
                    if i1==i and j1==j:
                        continue
                    n+=get(b,i1,j1)
            if b[i][j]:
                c=n==2 or n==3
            else:
                c=n==3
            row.append(int(c))
        b1.append(row)
    return b1

def printboard(b):
    for row in b:
        for c in row:
            print('@'if c else'.',end=' ')
        print()
    print()

b=randboard()
for i in range(5):
    printboard (b)
    b=step(b)
