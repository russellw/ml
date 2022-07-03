import math

print("x,y,z")
for y in range(10):
    y1 = y - 5
    for x in range(10):
        x1 = x - 5
        z = math.sin(math.sqrt(x1 ** 2 + y1 ** 2))
        print(f"{x},{y},{z}")
