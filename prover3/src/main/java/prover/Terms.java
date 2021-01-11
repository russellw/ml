package prover;

import java.util.*;

public final class Terms {
  private Terms() {}

  public static boolean match(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return false;

    // Variable
    if (a instanceof Variable) {
      var a1 = (Variable) a;

      // Existing mapping
      var a2 = map.get(a1);
      if (a2 != null) return a2.equals(b);

      // New mapping
      map.put(a1, b);
      return true;
    }

    // Compounds
    if (a instanceof List) {
      var a1 = (List) a;
      assert a1.get(0) != Symbol.EQUALS;
      if (b instanceof List) {
        var b1 = (List) b;
        assert b1.get(0) != Symbol.EQUALS;
        int n = a1.size();
        if (n != b1.size()) return false;
        if (a1.get(0) != b1.get(0)) return false;
        for (var i = 1; i < n; i++) if (!match(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

  public static boolean occurs(Variable a, Object b, Map<Variable, Object> map) {
    if (b instanceof Variable) {
      if (a == b) return true;
      var b1 = map.get(b);
      if (b1 != null) return occurs(a, b1, map);
      return false;
    }
    if (b instanceof List) {
      var b1 = (List) b;
      for (var x : b1) if (occurs(a, x, map)) return true;
    }
    return false;
  }

  private static boolean unifyVariable(Variable a, Object b, Map<Variable, Object> map) {
    // Existing mappings
    var a1 = map.get(a);
    if (a1 != null) return unify(a1, b, map);
    if (b instanceof Variable) {
      var b1 = map.get(b);
      if (b1 != null) return unify(a, b1, map);
    }

    // Occurs check
    if (occurs(a, b, map)) return false;

    // New mapping
    map.put(a, b);
    return true;
  }

  public static boolean unify(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return false;

    // Variable
    if (a instanceof Variable) return unifyVariable((Variable) a, b, map);
    if (b instanceof Variable) return unifyVariable((Variable) b, a, map);

    // Compounds
    if (a instanceof List) {
      var a1 = (List) a;
      if (b instanceof List) {
        var b1 = (List) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        if (a1.get(0) != b1.get(0)) return false;
        for (var i = 1; i < n; i++) if (!unify(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

  public static Object replace(Object a, Map<Variable, Object> map) {
    return Etc.treeMap(
        a,
        b -> {
          if (b instanceof Variable) {
            var b1 = map.get(b);
            if (b1 != null) return replace(b1, map);
          }
          return b;
        });
  }

  @SuppressWarnings("unchecked")
  private static void getFreeVariables(Set<Variable> bound, Object a, Set<Variable> r) {
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

  public static Set<Variable> freeVariables(Object a) {
    var r = new LinkedHashSet<Variable>();
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

  public static List<Object> implies(Object a, Object b) {
    return List.of(Symbol.OR, List.of(Symbol.NOT, a), b);
  }
}
