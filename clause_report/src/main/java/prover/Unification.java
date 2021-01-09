package prover;

import io.vavr.collection.Map;
import io.vavr.collection.Seq;

public final class Unification {
  private Unification() {}

  public static Map<Variable, Object> match(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return map;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return null;

    // Variable
    if (a instanceof Variable) {
      var a1 = (Variable) a;

      // Existing mapping
      var a2 = map.getOrElse(a1, null);
      if (a2 != null) return a2.equals(b) ? map : null;

      // New mapping
      return map.put(a1, b);
    }

    // Compounds
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      assert a1.head() != Symbol.EQUALS;
      if (b instanceof Seq) {
        var b1 = (Seq) b;
        assert b1.head() != Symbol.EQUALS;
        int n = a1.size();
        if (n != b1.size()) return null;
        for (var i = 0; i < n; i++) {
          map = match(a1.get(i), b1.get(i), map);
          if (map == null) return null;
        }
        return map;
      }
      return null;
    }

    // Atoms
    return a.equals(b) ? map : null;
  }

  private static boolean occurs(Variable a, Object b, Map<Variable, Object> map) {
    if (b instanceof Variable) {
      if (a == b) return true;
      var b1 = map.getOrElse((Variable) b, null);
      if (b1 != null) return occurs(a, b1, map);
      return false;
    }
    if (b instanceof Seq) {
      var b1 = (Seq) b;
      assert b1.head() != Symbol.EQUALS;
      for (var x : b1) if (occurs(a, x, map)) return true;
    }
    return false;
  }

  private static Map<Variable, Object> unifyVariable(
      Variable a, Object b, Map<Variable, Object> map) {
    // Existing mapping
    var a1 = map.getOrElse(a, null);
    if (a1 != null) return unify(a1, b, map);

    // Variable
    if (b instanceof Variable) {
      var b1 = map.getOrElse((Variable) b, null);
      if (b1 != null) return unify(b1, a, map);
    }

    // Occurs check
    if (occurs(a, b, map)) return null;

    // New mapping
    return map.put(a, b);
  }

  public static Map<Variable, Object> unify(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return map;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return null;

    // Variable
    if (a instanceof Variable) return unifyVariable((Variable) a, b, map);
    if (b instanceof Variable) return unifyVariable((Variable) b, a, map);

    // Compounds
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      assert a1.head() != Symbol.EQUALS;
      if (b instanceof Seq) {
        var b1 = (Seq) b;
        assert b1.head() != Symbol.EQUALS;
        int n = a1.size();
        if (n != b1.size()) return null;
        for (var i = 0; i < n; i++) {
          map = unify(a1.get(i), b1.get(i), map);
          if (map == null) return null;
        }
        return map;
      }
      return null;
    }

    // Atoms
    return a.equals(b) ? map : null;
  }

  public static Object replace(Object a, Map<Variable, Object> map) {
    return Etc.treeMap(
        a,
        b -> {
          if (b instanceof Variable) {
            var b1 = map.getOrElse((Variable) b, null);
            if (b1 != null) return replace(b1, map);
          }
          return b;
        });
  }
}
