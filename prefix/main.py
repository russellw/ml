from etc import *
from interpreter import *
from parse import *

v = parse("test.k")
for a in v:
    ev(a, {})
