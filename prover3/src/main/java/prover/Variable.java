package prover;

import java.util.*;

public final class Variable {
  public final Object type;
  public static final Map<Variable, String> names = new HashMap<>();

  public Variable(Object type) {
    this.type = type;
  }

  @SuppressWarnings("unchecked")
  private static void getFreeVariables(Set<Variable> bound, Object a, java.util.Set<Variable> r) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
          case EXISTS:
            {
              var binding = (List) a1.get(1);
              bound = new HashSet<>(bound);
              bound.addAll(binding);
              getFreeVariables(bound, a1.get(2), r);
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

  public static java.util.Set<Variable> freeVariables(Object a) {
    var r = new java.util.LinkedHashSet<Variable>();
    getFreeVariables(new HashSet<>(), a, r);
    return r;
  }

  public static Object quantify(Object a) {
    var variables = freeVariables(a);
    if (variables.isEmpty()) return a;
    return List.of(Symbol.ALL, List.copyOf(variables), a);
  }

  public static Object unquantify(Object a) {
    while (a instanceof List) {
      var a1 = (List) a;
      if (a1.get(0) != Symbol.ALL) break;
      a = a1.get(2);
    }
    return a;
  }

  public static boolean isomorphic(Object a, Object b, Map<Variable, Variable> map) {
    // Equal
    if (a == b) return true;

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
    if (a instanceof List) {
      var a1 = (List) a;
      if (b instanceof List) {
        var b1 = (List) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        for (var i = 0; i < n; i++) if (!isomorphic(a1.get(i), b1.get(i), map)) return false;
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
