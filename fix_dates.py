import sys
import re

filename=sys.argv[1]
if  filename[-4:]!='.csv':
    print(filename+': not a csv file')
    exit(1)

#get rid of non-ascii bytes
def bytes_from_file(filename, chunksize=8192):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break

bs=[]
for b in bytes_from_file(filename):
    bs.append(b)
with open(filename, "wb") as f:
    for b in bs:
        if b>126:
            continue
        if b<32 and (b!=10 and b!=13 and b!=9):
            continue
        a=b.to_bytes(1,'little')
        f.write(a)

#fix dates
n=0
with open(filename) as f: lines = f.readlines()
for i in range(len(lines)):
    s=lines[i]
    r=re.search(r'(\d\d)/(\d\d)/(\d\d\d\d)',s)
    if not r:
        continue
    s=s[:r.span()[0]]+r.group(3)+'-'+r.group(2)+'-'+r.group(1)+s[r.span()[1]:]
    lines[i]=s
    n+=1

open(filename,'w').writelines(lines)
print('fixed '+str(n)+' dates')
