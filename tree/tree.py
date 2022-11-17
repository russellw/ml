import os
import sys

for root, dirs, files in os.walk(sys.argv[1]):
    for f in files:
        #print(root + "/" + f)
        print(os.path.join(root , f))
