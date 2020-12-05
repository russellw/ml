import subprocess

from prover import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prformula(c):
    reset_var_names()
    pr("cnf")
    pr("(")

    # name
    pr(c.name)
    pr(", ")

    # role
    if hasattr(c, "role"):
        pr(c.role)
    else:
        pr("plain")
    pr(", ")

    # content
    a = c.term()
    prterm(a)

    # end
    print(").")


def do_file(filename):
    global attempted
    global solved

    # list file
    if os.path.splitext(filename)[1] == ".lst":
        for s in open(filename).readlines():
            do_file(s.strip())
        return

    # try to solve
    start = time.time()
    set_timeout()
    fname = os.path.basename(filename)
    problem = read_problem(filename)
    sys.stdout = open("cnf.p", "w")
    for c in problem.clauses:
        prformula(c)
    sys.stdout.close()


do_file(sys.argv[1])
