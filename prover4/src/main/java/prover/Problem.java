package prover;

import java.util.*;

public final class Problem {
  public final long startTime = System.currentTimeMillis();
  public long endTime;
  private final List<String> files = new ArrayList<>();
  public final List<String> header = new ArrayList<>();
  public SZS expected;
  public double rating = -1;
  public final List<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public final Set<Func> skolems = new LinkedHashSet<>();
  public final List<Clause> clauses = new ArrayList<>();
  public Superposition superposition;
  public Clause refutation;
  public SZS result;

  public String file() {
    return files.get(0);
  }

  public void add(String file, int includeDepth) {
    files.add("\t".repeat(includeDepth) + file);
  }

  public void solve(long timeout) {
    if (conjecture != null)
      formulas.add(
          new Formula(List.of(Symbol.NOT, conjecture.term()), Inference.NEGATE, conjecture));
    Types.inferTypes(formulas, clauses);
    new CNF(this);
    superposition = new Superposition();
    superposition.solve(this, startTime + timeout);
    if (conjecture != null)
      switch (result) {
        case Satisfiable:
          result = SZS.CounterSatisfiable;
          break;
        case Unsatisfiable:
          result = SZS.Theorem;
          break;
      }
    if (expected != null && result != expected)
      switch (result) {
        case Unsatisfiable:
        case Theorem:
          if (expected == SZS.ContradictoryAxioms) break;
        case Satisfiable:
        case CounterSatisfiable:
          throw new IllegalStateException(result + " != " + expected);
      }
    endTime = System.currentTimeMillis();
  }

  public static boolean solved(SZS szs) {
    switch (szs) {
      case Unsatisfiable:
      case ContradictoryAxioms:
      case Theorem:
      case CounterSatisfiable:
      case Satisfiable:
        return true;
    }
    return false;
  }
}
