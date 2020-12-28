import re


class Problem:
    def __init__(self, name):
        self.name = name


problems = {}

# e-cnf
e = open("e-cnf1.log").readlines()[2:]
e = [s.rstrip() for s in e]
i = 0
while 1:
    if i == len(e):
        break
    m = re.match(r"/mnt/c/ml/TPTP/Problems/.../(.*-.*)\.p", e[i])
    if not m:
        i += 1
        continue
    name = m[1]
    i += 1
    p = Problem(name)
    problems[name] = p
    if e[i] == "Timeout":
        p.timeout = 1
        i += 1
        continue
    while i < len(e) and not e[i].startswith("/mnt/c/ml/TPTP"):
        m = re.match(r"# SZS status (\w+)", e[i])
        if m:
            p.szs = m[1]
            i += 1
            continue
        m = re.match(r"# Proof object clause steps            : (\d+)", e[i])
        if m:
            p.e_clause_steps = int(m[1])
            i += 1
            continue
        m = re.match(r"# Proof object total steps             : (\d+)", e[i])
        if m:
            p.e_total_steps = int(m[1])
            i += 1
            continue
        m = re.match(r"# Proof object formula steps           : (\d+)", e[i])
        if m:
            p.e_formula_steps = int(m[1])
            i += 1
            continue
        m = re.match(r"# Proof object given clauses           : (\d+)", e[i])
        if m:
            p.e_gcgood = int(m[1])
            i += 1
            continue
        m = re.match(r"# Proof search given clauses           : (\d+)", e[i])
        if m:
            p.e_gctot = int(m[1])
            i += 1
            continue
        if (
            e[i] == "# SZS output start Saturation"
            or e[i] == "# SZS output start CNFRefutation"
        ):
            i += 1
            proof = []
            while (
                e[i] != "# SZS output end Saturation"
                and e[i] != "# SZS output end CNFRefutation"
            ):
                proof.append(e[i])
                i += 1
            i += 1
            p.e_proof = proof
            continue
        i += 1


# s-cnf
e = open("s-cnf.log").readlines()[2:]
e = [s.rstrip() for s in e]
i = 0
while 1:
    if i == len(e):
        break
    m = re.match(r"% SZS status Unsatisfiable for (.+).p", e[i])
    if not m:
        i += 1
        continue
    name = m[1]
    i += 1
    p = problems[name]
    proof = []
    while e[i]:
        proof.append(e[i])
        i += 1
    i += 1
    p.s_proof = proof

e = open("s-gc.log").readlines()[0:]
e = [s.rstrip() for s in e]
i = 0
while 1:
    if i == len(e):
        break
    m = e[i].split(",")
    name = m[0][:-2]
    i += 1
    p = problems[name]
    p.s_gcgood = int(m[1])
    p.s_gctot = int(m[2])


def p_key(p):
    return p.e_total_steps


ps = problems.values()
for p in ps:
    if hasattr(p, "s_proof") and not hasattr(p, "e_proof"):
        print(p.name)
    if not (
        hasattr(p, "e_clause_steps")
        == hasattr(p, "e_formula_steps")
        == hasattr(p, "e_total_steps")
        == hasattr(p, "e_proof")
    ):
        print(dir(p))
        raise ValueError(p.name)
ps = [p for p in ps if hasattr(p, "e_clause_steps")]
ps = [p for p in ps if hasattr(p, "s_gcgood")]
for p in ps:
    if p.e_total_steps != p.e_clause_steps + p.e_formula_steps:
        print(dir(p))
        raise ValueError(p.name)
    if len(p.e_proof) != p.e_total_steps:
        print(dir(p))
        raise ValueError(p.name)
ps = sorted(ps, key=p_key)


def writecsv(ps, filename):
    f = open(filename, "w")
    f.write("name")
    f.write(",")
    f.write("e_gcgood")
    f.write(",")
    f.write("e_gctot")
    f.write(",")
    f.write("e_gcratio")
    f.write(",")
    f.write("s_gcgood")
    f.write(",")
    f.write("s_gctot")
    f.write(",")
    f.write("s_gcratio")
    f.write(",")
    f.write("e_formula_steps")
    f.write(",")
    f.write("e_clause_steps")
    f.write(",")
    f.write("e_total_steps")
    f.write(",")
    f.write("s_proof")
    f.write("\n")
    for p in ps:
        f.write(p.name)
        f.write(",")
        f.write(str(p.e_gcgood))
        f.write(",")
        f.write(str(p.e_gctot))
        f.write(",")
        f.write(str(p.e_gcgood / p.e_gctot))
        f.write(",")
        f.write(str(p.s_gcgood))
        f.write(",")
        f.write(str(p.s_gctot))
        f.write(",")
        f.write(str(p.s_gcgood / p.s_gctot))
        f.write(",")
        f.write(str(p.e_formula_steps))
        f.write(",")
        f.write(str(p.e_clause_steps))
        f.write(",")
        f.write(str(p.e_total_steps))
        f.write(",")
        if hasattr(p, "s_proof"):
            f.write(str(len(p.s_proof)))
        else:
            f.write("0")
        f.write("\n")


writecsv([p for p in ps if p.szs == "Satisfiable"], "sat.csv")
writecsv([p for p in ps if p.szs == "Unsatisfiable"], "unsat.csv")
