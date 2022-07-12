import joy

size = 30

qs = []
while len(qs) < 100:
    q = joy.rand(size)
    try:
        joy.run(q, 0)
        qs.append(q)
        print(q)
    except:
        pass
print("-------------------------")


def good(q, x):
    try:
        r = joy.run(x, q)
        return joy.run(q, r)
    except:
        pass


def rating(x):
    try:
        joy.run(x, 0)
    except:
        return 0
    n = 0
    for q in qs:
        if good(q, x):
            n += 1
    return n


best = None
bestr = 0
for i in range(1000000):
    x = joy.rand(size)
    r = rating(x)
    if r > bestr:
        print()
        print(x)
        print(r)
        best = x
        bestr = r
