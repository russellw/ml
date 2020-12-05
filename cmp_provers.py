import subprocess

from prover import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prformula(c):
    reset_var_names()
    pr("fof")
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
    a = quantify(a)
    prterm(a)

    # end
    print(").")


def do_file(filename):
    for s in open(filename).readlines():
        m = re.match(r"%\s*Status\s*:\s*(\w+)", s)
        if m:
            print(s)

    # list file
    if os.path.splitext(filename)[1] == ".lst":
        for s in open(filename).readlines():
            do_file(s.strip())
        return

    # try to solve
    start = time.time()
    set_timeout()
    fname = os.path.basename(filename)
    try:
        problem = read_problem(filename)
        r, conclusion = solve(problem.clauses)
        if hasattr(problem, "conjecture"):
            if r == "Satisfiable":
                r = "CounterSatisfiable"
            elif r == "Unsatisfiable":
                r = "Theorem"
        print(f"prover.py says {r}")
    except (Inappropriate, RecursionError, Timeout) as e:
        print(f"prover.py says {e}")
    p = subprocess.Popen(
        ["bin/eprover", "-l", "0", "--generated-limit=100000", filename],
        stdout=subprocess.PIPE,
    )
    for s in p.stdout.readlines():
        s = str(s, "utf-8")
        m = re.match(r".*SZS status.*", s)
        if m:
            print("bin/eprover says")
            print(s)


do_file(sys.argv[1])
