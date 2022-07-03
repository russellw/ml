import mandelbrot

print("x,y,in")
v = mandelbrot.table()
for a in v:
    print(",".join(str(x) for x in a))
