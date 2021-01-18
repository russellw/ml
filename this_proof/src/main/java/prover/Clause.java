package prover;

import java.util.*;

public final class Clause extends AbstractFormula {
  public final Object[] literals;
  public final int negativeSize;
  public boolean subsumed;

  public Clause(
      List<Object> negative, List<Object> positive, Inference inference, AbstractFormula... from) {
    super(inference, from);

    // Simplify
    for (var i = 0; i < negative.size(); i++) negative.set(i, Terms.simplify(negative.get(i)));
    for (var i = 0; i < positive.size(); i++) positive.set(i, Terms.simplify(positive.get(i)));

    // Redundancy
    negative.removeIf(a -> a == Boolean.TRUE);
    positive.removeIf(a -> a == Boolean.FALSE);

    // Tautology
    for (var a : negative)
      if (a == Boolean.FALSE) {
        literals = new Object[] {true};
        negativeSize = 0;
        volume = calcVolume();
        return;
      }
    for (var a : positive)
      if (a == Boolean.TRUE) {
        literals = new Object[] {true};
        negativeSize = 0;
        volume = calcVolume();
        return;
      }
    for (var a : negative)
      for (var b : positive)
        if (a.equals(b)) {
          literals = new Object[] {true};
          negativeSize = 0;
          volume = calcVolume();
          return;
        }

    // Literals
    literals = new Object[negative.size() + positive.size()];
    for (var i = 0; i < negative.size(); i++) literals[i] = negative.get(i);
    for (var i = 0; i < positive.size(); i++) literals[negative.size() + i] = positive.get(i);
    negativeSize = negative.size();
    volume = calcVolume();
  }

  private int calcVolume() {
    int[] n = new int[1];
    Etc.walkLeaves(Arrays.asList(literals), a -> n[0]++);
    if (Main.memo != null) {
      var d = renameVariables();
      var i = 0;
      for (var c : Main.memo) {
        if (Isomorphism.subsumes(d, c)) {
          n[0] -= 1000000;
          n[0] += i * 1000;
          break;
        }
        i++;
      }
    }
    return n[0];
  }

  public Set<Variable> variables() {
    var r = new HashSet<Variable>();
    Etc.walkLeaves(
        Arrays.asList(literals),
        a -> {
          if (a instanceof Variable) r.add((Variable) a);
        });
    return r;
  }

  private Clause(
      Object[] literals,
      int volume,
      int negativeSize,
      Inference inference,
      AbstractFormula[] from) {
    super(inference, from);
    this.literals = literals;
    this.negativeSize = negativeSize;
    this.volume = volume;
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
            Etc.mapLeaves(
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
    return new Clause(
        r.toArray(), volume, negativeSize, Inference.RENAME_VARIABLES, new Clause[] {this});
  }

  public Clause original() {
    if (inference == Inference.RENAME_VARIABLES) return (Clause) from[0];
    return this;
  }

  public final int volume;

  public int volume() {
    return volume;
  }

  @Override
  public String toString() {
    return Arrays.toString(negative()) + " => " + Arrays.toString(positive());
  }

  public Object[] negative() {
    return Arrays.copyOf(literals, negativeSize);
  }

  public Object[] positive() {
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
    return r;
  }
}
