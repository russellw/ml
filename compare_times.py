import time
import torch

iterations = 10000
size=1000

########################################
def f():
    row=[0]*size
    a=[]
    for i in range(size):
        a.append([0]*size)
    for i in range(iterations):
        a[10][10]=1-a[10][10]


start = time.time()
f()
print(f"{time.time()-start:12.6f}")

########################################
def f():
    row=[0]*size
    a=[]
    for i in range(size):
        a.append([0]*size)
    a=torch.tensor(a)
    for i in range(iterations):
        a[10][10]=1-a[10][10]


start = time.time()
f()
print(f"{time.time()-start:12.6f}")
