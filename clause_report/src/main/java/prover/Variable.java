package prover;

import io.vavr.collection.Array;
import io.vavr.collection.HashSet;
import io.vavr.collection.Seq;
import io.vavr.collection.Set;
import java.util.HashMap;

public final class Variable {
  public final Object type;
  public static final HashMap<Variable, String> names = new HashMap<>();

  public Variable(Object type) {
    this.type = type;
  }

  @SuppressWarnings("unchecked")
  private static void getFreeVariables(
      Set<Variable> bound, Object a, java.util.HashSet<Variable> r) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
          case EXISTS:
            {
              var binding = (Seq) a1.get(1);
              getFreeVariables(bound.addAll(binding), a1.get(2), r);
              return;
            }
        }
      for (var b : a1) getFreeVariables(bound, b, r);
      return;
    }
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      if (!bound.contains(a1)) r.add(a1);
    }
  }

  public static java.util.HashSet<Variable> freeVariables(Object a) {
    var r = new java.util.LinkedHashSet<Variable>();
    getFreeVariables(HashSet.empty(), a, r);
    return r;
  }

  public static Object quantify(Object a) {
    var variables = freeVariables(a);
    if (variables.isEmpty()) return a;
    return Array.of(Symbol.ALL, Array.ofAll(variables), a);
  }

  public static Object unquantify(Object a) {
    while (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.head() != Symbol.ALL) break;
      a = a1.get(2);
    }
    return a;
  }

  public static boolean isomorphic(Object a, Object b, HashMap<Variable, Variable> map) {
    // Equal
    if (a == b) return true;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return false;

    // Variable
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      if (b instanceof Variable) {
        var b1 = (Variable) b;
        var a2 = map.get(a1);
        var b2 = map.get(b1);

        // Compatible mapping
        if (a1 == b2 && b1 == a2) return true;

        // New mapping
        if (a2 == null && b2 == null) {
          map.put(a1, b1);
          map.put(b1, a1);
          return true;
        }
      }
      return false;
    }

    // Compounds
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (b instanceof Seq) {
        var b1 = (Seq) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        if (a1.head() != b1.head()) return false;
        for (var i = 1; i < n; i++) if (!isomorphic(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

  @Override
  public String toString() {
    var name = names.get(this);
    if (name == null) {
      var i = names.size();
      name = i < 26 ? Character.toString('A' + i) : "Z" + (i - 25);
      names.put(this, name);
    }
    return name;
  }
}
