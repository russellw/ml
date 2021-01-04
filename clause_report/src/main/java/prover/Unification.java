package prover;

import java.util.Map;

public final class Unification {
  public static boolean match(Term a, Term b, Map<Variable, Term> map) {
    assert !(a instanceof Eq);
    assert !(b instanceof Eq);

    // Equal
    if (a == b) return true;

    // Type mismatch
    if (a.isBoolean() != b.isBoolean()) return false;

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

    // Atom
    int n = a.size();
    if (n == 0) return false;

    // Compound
    if (n != b.size()) return false;
    for (var i = 0; i < n; i++) if (!match(a.get(i), b.get(i), map)) return false;
    return true;
  }

  public static boolean unify(Term a, Term b, Map<Variable, Term> map) {
    assert !(a instanceof Eq);
    assert !(b instanceof Eq);

    // Equal
    if (a == b) return true;

    // Type mismatch
    if (a.isBoolean() != b.isBoolean()) return false;

    // Variable
    if (a instanceof Variable) return unifyVariable((Variable) a, b, map);
    if (b instanceof Variable) return unifyVariable((Variable) b, a, map);

    // Atom
    int n = a.size();
    if (n == 0) return false;

    // Compound
    if (n != b.size()) return false;
    for (var i = 0; i < n; i++) if (!unify(a.get(i), b.get(i), map)) return false;
    return true;
  }

  private static boolean unifyVariable(Variable a, Term b, Map<Variable, Term> map) {
    // Equal
    if (a == b) return true;

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

  private static boolean occurs(Variable a, Term b, Map<Variable, Term> map) {
    if (a == b) return true;
    if (b instanceof Variable) {
      var b1 = map.get(b);
      if (b1 != null) return occurs(a, b1, map);
      return false;
    }
    for (var bi : b) if (occurs(a, bi, map)) return true;
    return false;
  }
}
