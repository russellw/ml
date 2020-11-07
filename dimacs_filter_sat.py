import argparse

parser = argparse.ArgumentParser(
    description="filter list of DIMACS files by satisfiable"
)
parser.add_argument("list_file")
args = parser.parse_args()


def read_lines(filename):
    with open(filename) as f:
        return [s.rstrip("\n") for s in f]


class Dimacs:
    def __init__(self, filename):
        self.filename = filename
        self.expected_sat = None
        with open(filename) as f:
            for s in f:
                if s[0] == "c":
                    if "UNSAT" in s:
                        self.expected_sat = False
                        break
                    if "SAT" in s:
                        self.expected_sat = True
                        break


files = read_lines(args.list_file)
problems = [Dimacs(s) for s in files]
for p in problems:
    if p.expected_sat is True:
        print(p.filename)
