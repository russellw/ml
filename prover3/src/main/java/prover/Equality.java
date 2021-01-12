package prover;

import java.util.List;

// Normally, first-order logic does not allow equality of predicates (Boolean terms)
// However, superposition calculus takes the view that a predicate is an equation
// p=true
public final class Equality {
  private Equality() {}

  public static Object of(Object a, Object b) {
    if (!equatable(a, b)) throw new IllegalArgumentException(a.toString() + '=' + b);
    if (b == Boolean.TRUE) return a;
    return List.of(Symbol.EQUALS, a, b);
  }

  public static boolean equatable(Object a, Object b) {
    var type = Types.typeof(a);
    if (!type.equals(Types.typeof(b))) return false;
    return type != Symbol.BOOLEAN || b == Boolean.TRUE;
  }

  public static Object left(Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      if (a1.get(0) == Symbol.EQUALS) return a1.get(1);
    }
    return a;
  }

  public static Object right(Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      if (a1.get(0) == Symbol.EQUALS) return a1.get(2);
    }
    return true;
  }
}
