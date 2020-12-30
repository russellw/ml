package prover;

import java.util.*;

public final class Clause {
  public final Term[] literals;
  public final int negativeSize;
  public boolean subsumed;

  private static void setBoolean(Term a) {
    if (a instanceof Function) {
      ((Function) a).isBoolean = true;
      return;
    }
    if (a instanceof Call) {
      var a1 = (Call) a;
      ((Function) a1.get(0)).isBoolean = true;
      return;
    }
  }

  public Clause(List<Term> negative, List<Term> positive) {
    // Types
    for (var a : negative) setBoolean(a);
    for (var a : positive) setBoolean(a);

    // Simplify
    for (var i = 0; i < negative.size(); i++) negative.set(i, simplify(negative.get(i)));
    for (var i = 0; i < positive.size(); i++) positive.set(i, simplify(positive.get(i)));

    // Redundancy
    negative.removeIf(a -> a == Term.TRUE);
    positive.removeIf(a -> a == Term.FALSE);

    // Tautology
    for (var a : negative) {
      if (a == Term.FALSE) {
        literals = new Term[] {Term.TRUE};
        negativeSize = 0;
        return;
      }
    }
    for (var a : positive) {
      if (a == Term.TRUE) {
        literals = new Term[] {Term.TRUE};
        negativeSize = 0;
        return;
      }
    }
    for (var a : negative) {
      for (var b : positive) {
        if (a.equals(b)) {
          literals = new Term[] {Term.TRUE};
          negativeSize = 0;
          return;
        }
      }
    }

    // Literals
    literals = new Term[negative.size() + positive.size()];
    for (var i = 0; i < negative.size(); i++) literals[i] = negative.get(i);
    for (var i = 0; i < positive.size(); i++) literals[negative.size() + i] = positive.get(i);
    negativeSize = negative.size();
  }

  private static Term simplify(Term a) {
    if (a instanceof Equation) {
      var x = a.get(0);
      var y = a.get(1);
      if (x.equals(y)) return Term.TRUE;
    }
    return a;
  }

  private static void getVariables(Term a, Set<Variable> r) {
    if (a instanceof Variable) {
      r.add((Variable) a);
      return;
    }
    for (var b : a) getVariables(b, r);
  }

  public Set<Variable> variables() {
    var r = new HashSet<Variable>();
    for (var a : literals) getVariables(a, r);
    return r;
  }

  private Clause(Term[] literals, int negativeSize) {
    this.literals = literals;
    this.negativeSize = negativeSize;
  }

  public final boolean isFalse() {
    return literals.length == 0;
  }

  public final boolean isTrue() {
    return (literals.length == 1) && (negativeSize == 0) && (literals[0] == Term.TRUE);
  }

  public final Term[] negative() {
    return Arrays.copyOf(literals, negativeSize);
  }

  public final Term[] positive() {
    return Arrays.copyOfRange(literals, negativeSize, literals.length);
  }

  public final int positiveSize() {
    return literals.length - negativeSize;
  }

  public Clause rename() {
    var map = new HashMap<Variable, Variable>();
    var literals = Term.transform(this.literals, term -> term.rename(map));
    if (map.isEmpty()) {
      return this;
    }
    return new Clause(literals, negativeSize);
  }

  @Override
  public String toString() {
    return Arrays.toString(negative()) + " => " + Arrays.toString(positive());
  }

  public int volume() {
    int n = 0;
    for (var a : literals) {
      n += volume(a);
    }
    return n;
  }

  private static int volume(Term a) {
    int n = 1;
    for (var b : a) {
      n += volume(b);
    }
    return n;
  }
}
