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
    print(f"% {filename}")
    try:
        problem = read_problem(filename)
        if problem.formulas:
            print(f"% {len(problem.formulas)} formulas")
        print(f"% {len(problem.clauses)} clauses")
        r, conclusion = solve(problem.clauses)
        if hasattr(problem, "conjecture"):
            if r == "Satisfiable":
                r = "CounterSatisfiable"
            elif r == "Unsatisfiable":
                r = "Theorem"
        print(f"% SZS status {r} for {fname}")
        if conclusion:
            for c in conclusion.proof():
                prformula(c)
        if r in (
            "Theorem",
            "Unsatisfiable",
            "ContradictoryAxioms",
            "Satisfiable",
            "CounterSatisfiable",
        ):
            if problem.expected and r != problem.expected:
                if problem.expected == "ContradictoryAxioms" and r in (
                    "Theorem",
                    "Unsatisfiable",
                ):
                    pass
                else:
                    print(f"{r} != {problem.expected}")
    except (Inappropriate, RecursionError, Timeout) as e:
        print(f"% SZS status {e} for {fname}")
    print(f"% {time.time() - start:.3f} seconds")
    print()
    for c in conclusion.proof():
        if c.parents:
            if c.inference == "negate":
                continue
            sys.stdout = open("1.p", "w")
            for d in c.parents:
                d.role = "axiom"
                prformula(d)
            c.role = "conjecture"
            prformula(c)
            sys.stdout.close()
            p = subprocess.Popen(["bin/eprover", "1.p"], stdout=subprocess.PIPE)
            s = p.stdout.read()
            s = str(s, "utf-8")
            if "SZS status Theorem" in s:
                continue
            eprint(s)
            break


do_file(sys.argv[1])
