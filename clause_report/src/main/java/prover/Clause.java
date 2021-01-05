package prover;

import io.vavr.collection.Array;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public final class Clause {
  private final Object[] literals;
  public final int negativeSize;
  public boolean subsumed;

  public int size() {
    return literals.length;
  }

  public Object get(int i) {
    return literals[i];
  }

  public Seq<Object> literals() {
    return Array.of(literals);
  }

  private static void setBoolean(Object a) {
    if (a instanceof Seq) a = ((Seq) a).head();
    if (a instanceof Func) ((Func) a).isBoolean = true;
  }

  public Clause(ArrayList<Object> negative, ArrayList<Object> positive) {
    // Types
    for (var a : negative) setBoolean(a);
    for (var a : positive) setBoolean(a);

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
    Etc.treeForEach(
        literals(),
        a -> {
          if (a instanceof Variable) r.add((Variable) a);
        });
    return r;
  }

  private Clause(Object[] literals, int negativeSize) {
    this.literals = literals;
    this.negativeSize = negativeSize;
  }

  public final boolean isFalse() {
    return literals.length == 0;
  }

  public final boolean isTrue() {
    return (literals.length == 1) && (negativeSize == 0) && (literals[0] == Boolean.TRUE);
  }

  public final Seq<Object> negative() {
    return literals().slice(0, negativeSize);
  }

  public final Seq<Object> positive() {
    return literals().slice(negativeSize, literals.length);
  }

  public final int positiveSize() {
    return literals.length - negativeSize;
  }

  public Clause renameVariables() {
    var map = new HashMap<Variable, Variable>();
    var r =
        Etc.treeMap(
            literals(),
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
    return new Clause(((Seq) r).toJavaArray(), negativeSize);
  }

  @Override
  public String toString() {
    return negative() + " => " + positive();
  }

  public int volume() {
    int[] n = new int[1];
    Etc.treeForEach(literals(), a -> n[0]++);
    return n[0];
  }
}
