import sys

out = open("a", "wb")
for c in open(sys.argv[1], "rb").read():
    if c < 127:
        out.write(c.to_bytes(1, byteorder="big"))
