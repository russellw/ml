package prover;

import java.util.List;
import java.util.Map;

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
    // Existing mapping
    var a1 = map.get(a);
    if (a1 != null) return unify(a1, b, map);

    // Variable
    if (b instanceof Variable) {
      var b1 = map.get(b);
      if (b1 != null) return unify(b1, a, map);
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
}
