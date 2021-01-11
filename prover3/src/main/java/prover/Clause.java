package prover;

import java.util.*;

public final class Clause extends AbstractFormula {
  public final Object[] literals;
  public final int negativeSize;
  public boolean subsumed;

  public int size() {
    return literals.length;
  }

  public int negativeSize() {
    return negativeSize;
  }

  public Object get(int i) {
    return literals[i];
  }

  public Clause(
      List<Object> negative, List<Object> positive, Inference inference, AbstractFormula... from) {
    super(inference, from);

    // Redundancy
    negative.removeIf(a -> a == Boolean.TRUE);
    positive.removeIf(a -> a == Boolean.FALSE);

    // Tautology
    for (var a : negative) {
      if (a == Boolean.FALSE) {
        literals = new Object[] {true};
        negativeSize = 0;
        return;
      }
    }
    for (var a : positive) {
      if (a == Boolean.TRUE) {
        literals = new Object[] {true};
        negativeSize = 0;
        return;
      }
    }

    // Literals
    literals = new Object[negative.size() + positive.size()];
    for (var i = 0; i < negative.size(); i++) literals[i] = negative.get(i);
    for (var i = 0; i < positive.size(); i++) literals[negative.size() + i] = positive.get(i);
    negativeSize = negative.size();
  }

  public HashSet<Variable> variables() {
    var r = new HashSet<Variable>();
    Etc.treeWalk(
        Arrays.asList(literals),
        a -> {
          if (a instanceof Variable) r.add((Variable) a);
        });
    return r;
  }

  private Clause(Object[] literals, int negativeSize, Inference inference, AbstractFormula[] from) {
    super(inference, from);
    this.literals = literals;
    this.negativeSize = negativeSize;
  }

  public final boolean isFalse() {
    return literals.length == 0;
  }

  public final boolean isTrue() {
    return (literals.length == 1) && (negativeSize == 0) && (literals[0] == Boolean.TRUE);
  }

  public final int positiveSize() {
    return literals.length - negativeSize;
  }

  public Clause renameVariables() {
    var map = new HashMap<Variable, Variable>();
    var r =
        (List)
            Etc.treeMap(
                Arrays.asList(literals),
                a -> {
                  if (a instanceof Variable) {
                    var a1 = (Variable) a;
                    var b = map.get(a1);
                    if (b == null) {
                      b = new Variable(a1.type);
                      map.put(a1, b);
                    }
                    return b;
                  }
                  return a;
                });
    if (map.isEmpty()) return this;
    return new Clause(r.toArray(), negativeSize, Inference.RENAME_VARIABLES, new Clause[] {this});
  }

  public Clause original() {
    if (inference == Inference.RENAME_VARIABLES) return (Clause) from[0];
    return this;
  }

  public int volume() {
    int[] n = new int[1];
    Etc.treeWalk(Arrays.asList(literals), a -> n[0]++);
    return n[0];
  }

  @Override
  public String toString() {
    return Arrays.toString(negative()) + " => " + Arrays.toString(positive());
  }

  public final Object[] negative() {
    return Arrays.copyOf(literals, negativeSize);
  }

  public final Object[] positive() {
    return Arrays.copyOfRange(literals, negativeSize, literals.length);
  }

  @Override
  public Object term() {
    var r = new ArrayList<>();
    r.add(Symbol.OR);
    for (var i = 0; i < negativeSize; i++) r.add(List.of(Symbol.NOT, literals[i]));
    r.addAll(List.of(literals).subList(negativeSize, literals.length));
    switch (r.size()) {
      case 1:
        return false;
      case 2:
        return r.get(1);
    }
    return Etc.same(r);
  }
}
