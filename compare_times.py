import time

n = 10000000

########################################
def f():
    s = [1, 2, 3]
    for i in range(n):
        tuple(map(lambda x: x + 1, s))


start = time.time()
f()
print(f"{time.time()-start:12.6f}")

########################################
def f():
    s = [1, 2, 3]
    for i in range(n):
        r = []
        for x in s:
            r.append(x + 1)
        tuple(r)


start = time.time()
f()
print(f"{time.time()-start:12.6f}")
