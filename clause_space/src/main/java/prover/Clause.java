package prover;

import java.util.*;

public final class Clause {
  public final Term[] literals;
  public final int negativeSize;
  public boolean subsumed;

  private static void setBoolean(Term a) {
    if (a instanceof Func) {
      ((Func) a).isBoolean = true;
      return;
    }
    if (a instanceof Call) {
      var a1 = (Call) a;
      ((Func) a1.get(0)).isBoolean = true;
      return;
    }
  }

  private static void image(Term a, StringBuilder sb) {
    if (a instanceof Var) {
      sb.append('?');
      return;
    }
    if (a instanceof Call) {
      sb.append(a.get(0));
      sb.append('(');
      for (var i = 1; i < a.size(); i++) {
        if (i > 1) sb.append(',');
        image(a.get(i), sb);
      }
      sb.append(')');
      return;
    }
    if (a instanceof Eq) {
      var a1 = (Eq) a;
      image(a1.left, sb);
      sb.append('=');
      image(a1.right, sb);
      return;
    }
    sb.append(a);
  }

  public String image() {
    var sb = new StringBuilder();
    for (var i = 0; i < literals.length; i++) {
      if (i > 0) sb.append('|');
      if (i < negativeSize) sb.append('~');
      image(literals[i], sb);
    }
    return sb.toString();
  }

  public Clause(List<Term> negative, List<Term> positive) {
    // Types
    for (var a : negative) setBoolean(a);
    for (var a : positive) setBoolean(a);

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

    // Literals
    literals = new Term[negative.size() + positive.size()];
    for (var i = 0; i < negative.size(); i++) literals[i] = negative.get(i);
    for (var i = 0; i < positive.size(); i++) literals[negative.size() + i] = positive.get(i);
    negativeSize = negative.size();
  }

  private static void getVars(Term a, Set<Var> r) {
    if (a instanceof Var) {
      r.add((Var) a);
      return;
    }
    for (var b : a) getVars(b, r);
  }

  public Set<Var> vars() {
    var r = new HashSet<Var>();
    for (var a : literals) getVars(a, r);
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

  private static Term renameVars(Term a, Map<Var, Var> map) {
    if (a instanceof Var) {
      var a1 = (Var) a;
      var b = map.get(a1);
      if (b == null) {
        b = new Var();
        map.put(a1, b);
      }
      return b;
    }
    return a.transform(b -> renameVars(b, map));
  }

  public Clause renameVars() {
    var map = new HashMap<Var, Var>();
    var literals = Term.transform(this.literals, a -> renameVars(a, map));
    if (map.isEmpty()) return this;
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
