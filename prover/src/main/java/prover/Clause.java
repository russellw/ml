package prover;

import java.util.*;
import java.util.function.BiConsumer;

public final class Clause {
  public static Term cost;
  public final Clause[] from;
  public final Term[] literals;
  public final int negativeSize;
  public final boolean renamed;
  public boolean subsumed;

  public Clause(List<Term> negative, List<Term> positive, Clause... from) {
    this.from = originals(from);
    renamed = false;
    type(negative);
    type(positive);

    // Simplify
    for (var i = 0; i < negative.size(); i++) {
      negative.set(i, negative.get(i).simplify());
    }
    for (var i = 0; i < positive.size(); i++) {
      positive.set(i, positive.get(i).simplify());
    }

    // Redundancy
    negative.removeIf(a -> a == Term.TRUE);
    positive.removeIf(a -> a == Term.FALSE);

    // Tautology?
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
    for (var i = 0; i < negative.size(); i++) {
      literals[i] = negative.get(i);
    }
    for (var i = 0; i < positive.size(); i++) {
      literals[negative.size() + i] = positive.get(i);
    }
    negativeSize = negative.size();
  }

  private Clause(Term[] literals, int negativeSize, Clause... from) {
    this.from = from;
    renamed = true;
    this.literals = literals;
    this.negativeSize = negativeSize;
  }

  public double cost() {
    if (cost == null) {
      return volume();
    }
    return cost.eval(map()).number();
  }

  public final boolean isFalse() {
    return literals.length == 0;
  }

  public final boolean isTrue() {
    return (literals.length == 1) && (negativeSize == 0) && (literals[0] == Term.TRUE);
  }

  public HashMap<Variable, Term> map() {
    var map = new HashMap<Variable, Term>();
    for (var x : Function.COMMON_NAMES.values()) {
      map.put(x, new Number(0));
    }
    for (var a : literals) {
      a.walk(
          term -> {
            if (term instanceof Function) {
              var function = (Function) term;
              var x = function.variable;
              if (x != null) {
                map.put(x, new Number(map.get(x).number() + 1));
              }
            }
          });
    }
    return map;
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
    return new Clause(literals, negativeSize, this);
  }

  @Override
  public String toString() {
    return Arrays.toString(negative()) + " => " + Arrays.toString(positive());
  }

  public void walk(BiConsumer<Clause, Integer> f) {
    walk(f, 0);
  }

  private static Clause[] originals(Clause[] from) {
    var r = new Clause[from.length];
    for (int i = 0; i < from.length; i++) {
      var c = from[i];
      r[i] = c.renamed ? c.from[0] : c;
    }
    return r;
  }

  private static void type(List<Term> literals) {
    for (var a : literals) {
      switch (a.tag()) {
        case CALL:
        case EQ:
        case FALSE:
        case FUNCTION:
        case TRUE:
          a.type(Type.BOOLEAN);
          break;
        default:
          throw new IllegalArgumentException(a.toString());
      }
    }
  }

  private int volume() {
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

  private void walk(BiConsumer<Clause, Integer> f, int distance) {
    for (var c : from) {
      c.walk(f, distance + 1);
    }
    f.accept(this, distance);
  }
}
