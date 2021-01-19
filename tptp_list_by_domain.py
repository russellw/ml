import os

for root, dirs, files in os.walk(r"\TPTP\Problems"):
    if files:
        r = []
        for fname in files:
            filename = os.path.join(root, fname)
            if os.path.splitext(filename)[1] != ".p":
                continue
            if "^" in filename:
                continue
            r.append(filename + "\n")
        f = open(root[-3:] + ".lst", "w")
        f.writelines(r)
