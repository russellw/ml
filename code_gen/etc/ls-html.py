# Show the structure of a complex directory tree
# along several different dimensions
# in the form of a HTML report

import hashlib
import sys
import os
from datetime import datetime

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

print("<!DOCTYPE html>")
print('<html lang="en">')
print('<meta charset="utf-8"/>')
print("<style>")
print("td {")
print("padding-right: 10px;")
print("white-space: nowrap;")
print("}")
print("</style>")
print("<title>%s</title>\n" % os.getcwd())


class Dir:
    def __init__(self, s):
        self.s = s
        self.subdirs = {}

    def __repr__(self):
        return self.s

    def count(self):
        n = 1
        for d in self.subdirs.values():
            n += d.count()
        return n


def splitPath(s):
    if len(s) > 2 and s[0].isalpha() and s[1] == ":":
        s = s[2:]
    if s[0] == os.sep:
        s = s[1:]
    return s.split(os.sep)


dprefix = splitPath(os.getcwd())
droot = Dir(os.getcwd())


def mkDir(s):
    p = splitPath(s)
    i = 0
    while i < len(dprefix):
        assert dprefix[i] == p[i]
        i += 1
    d = droot
    while i < len(p):
        s = p[i]
        i += 1
        if s not in d.subdirs:
            d.subdirs[s] = Dir(s)
        d = d.subdirs[s]
    return d


extd = {}
exts = []


class Ext:
    def __init__(self, s):
        self.s = s
        extd[s] = self
        exts.append(self)
        self.fs = []

    def __repr__(self):
        return self.s


def mkExt(s):
    if s not in extd:
        extd[s] = Ext(s)
    return extd[s]


def hashFile(path):
    with open(path, "rb") as f:
        h = hashlib.sha512()
        blocksize = 1 << 20
        while 1:
            b = f.read(blocksize)
            if not b:
                return h.hexdigest()
            h.update(b)


def ref(s):
    s = s.replace("\\", "/")
    return s.replace(" ", "")


class File:
    def __init__(self, directory, name):
        mkDir(directory)
        self.directory = directory
        self.name = name
        self.tm = os.path.getmtime(os.path.join(directory, name))
        self.size = os.path.getsize(os.path.join(directory, name))
        basename, ext = os.path.splitext(name)
        if ext == "":
            ext = "none"
        self.ext = mkExt(ext)
        self.ext.fs.append(self)
        self.hsh = hashFile(os.path.join(directory, name))

    def __repr__(self):
        return self.name

    def link(self):
        return "<a href=file:///%s>%s</a>" % (
            ref(os.path.join(self.directory, self.name)),
            self.name,
        )


def h1(s):
    print('<h1 id="%s">%s</h1>' % (ref(s), s))


def h2(s):
    print('<h2 id="%s">%s</h2>' % (ref(s), s))


def href(s):
    return '<a href="#%s">%s</a>' % (ref(s), s)


def strtm(tm):
    return datetime.fromtimestamp(f.tm).strftime("%Y-%m-%d<small> %H:%M:%S</small>")


fs = []
for root, dirs, files in os.walk(os.getcwd()):
    if os.sep + ".git" + os.sep in root:
        continue
    if root.endswith(os.sep + ".git"):
        continue
    for name in files:
        fs.append(File(root, name))
exts.sort(key=lambda e: (-len(e.fs), e.s))

dups = {}
for f in fs:
    s = f.name
    if s not in dups:
        dups[s] = []
    dups[s].append(f)
dups1 = sorted(dups.keys())
dups1 = [s for s in dups1 if len(dups[s]) > 1]

# in principle, detection of duplicate files is fallible
# because we only check for same hash, not same contents
# if SHA-512 can be taken as effectively random
# then the probability of a hash collision
# is small compared to the probability of incorrectly detecting same contents
# because of cosmic rays flipping bits in the CPU
hdups = {}
for f in fs:
    s = f.hsh
    if s not in hdups:
        hdups[s] = []
    hdups[s].append(f)
hdups1 = sorted(hdups.keys())
hdups1 = [s for s in hdups1 if len(hdups[s]) > 1]

print("<table>")
print("<tr>")
print("<td>" + href("Directory tree"))
print(f'<td style="text-align: right">{len(droot.subdirs):,}')
print(f'<td style="text-align: right">{droot.count():,}')
print("<tr>")
print("<td>" + href("Extensions"))
print(f'<td style="text-align: right">{len(exts):,}')
print(f'<td style="text-align: right">{len(fs):,}')
if dups1:
    print("<tr>")
    print("<td>" + href("Duplicate names"))
    print(f'<td style="text-align: right">{len(dups1):,}')
    n = 0
    for z in dups.values():
        if len(z) > 1:
            n += len(z)
    print(f'<td style="text-align: right">{n:,}')
if hdups1:
    print("<tr>")
    print("<td>" + href("Duplicate files"))
    print(f'<td style="text-align: right">{len(hdups1):,}')
    n = 0
    for z in hdups.values():
        if len(z) > 1:
            n += len(z)
    print(f'<td style="text-align: right">{n:,}')
print("</table>")

h1("Directory tree")


def printDir(d):
    print("<ul>")
    for s in sorted(d.subdirs.keys()):
        print("<li>")
        print(s)
        printDir(d.subdirs[s])
        print("</li>")
    print("</ul>")


print(os.getcwd())
printDir(droot)

h1("Extensions")
print("<table>")
for e in exts:
    print("<tr>")
    print("<td>" + str(len(e.fs)))
    print("<td>" + href(e.s))
print("</table>")

for e in exts:
    h2(e.s)
    print("<table>")
    e.fs.sort(key=lambda f: (f.directory, f.name, f.tm))
    for f in e.fs:
        print("<tr>")
        print("<td>" + f.directory)
        print("<td>" + f.link())
        print("<td>" + strtm(f.tm))
        print(f'<td style="text-align: right">{f.size:,}')
    print("</table>")

if dups1:
    h1("Duplicate names")
    print("<table>")
    for s in dups1:
        a = dups[s]
        for i in range(len(a)):
            print("<tr>")
            f = a[i]
            print("<td>" + f.link())
            print("<td>" + f.directory)
            print("<td>" + strtm(f.tm))
            print(f'<td style="text-align: right">{f.size:,}')
    print("</table>")

if hdups1:
    h1("Duplicate files")
    print("<table>")
    for s in hdups1:
        a = hdups[s]
        for i in range(len(a)):
            print("<tr>")
            f = a[i]
            print("<td>" + f.directory)
            print("<td>" + f.link())
            print("<td>" + strtm(f.tm))
            if i == 0:
                print(f'<td style="text-align: right">{f.size:,}')
            else:
                print(f'<td style="text-align: right">')
    print("</table>")
