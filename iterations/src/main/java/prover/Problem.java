package prover;

import java.util.*;

public final class Problem {
  public final String file;
  public long startTime = System.currentTimeMillis();
  int iterations;
  public final Map<String, Func> funcs = new HashMap<>();
  public SZS expected;
  public double rating = -1;
  public final List<Formula> formulas = new ArrayList<>();
  public Formula conjecture;
  public final List<Clause> clauses = new ArrayList<>();
  public Clause refutation;
  public SZS result;

  public Problem(String file) {
    this.file = file;
  }

  public void solve(int clauseLimit, long timeout) {
    if (conjecture != null)
      formulas.add(
          new Formula(List.of(Symbol.NOT, conjecture.term()), Inference.NEGATE, conjecture));
    Types.inferTypes(formulas, clauses);
    new CNF(this);
    new Superposition(this, clauseLimit, timeout == 0 ? Long.MAX_VALUE : startTime + timeout);
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
  }

  public void solve2(int clauseLimit, long timeout) {
    for (var c : clauses) c.subsumed = false;
    new Superposition(this, clauseLimit, timeout == 0 ? Long.MAX_VALUE : startTime + timeout);
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
