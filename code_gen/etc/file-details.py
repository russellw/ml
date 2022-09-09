import sys
import os
from datetime import datetime

for s in sys.stdin:
    s = s.split("\t")[0].strip()
    st = os.stat(s)
    t = datetime.fromtimestamp(st.st_mtime)
    print(s + "\t" + str(t) + "\t" + str(st.st_size))
