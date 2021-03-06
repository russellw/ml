package prover;

import io.vavr.collection.Array;
import io.vavr.collection.Seq;

public final class Equality {
  private Equality() {}

  public static Object of(Object a, Object b) {
    if (!equatable(a, b)) throw new IllegalArgumentException(a.toString() + '=' + b);
    if (b == Boolean.TRUE) return a;
    return Array.of(Symbol.EQUALS, a, b);
  }

  public static boolean equatable(Object a, Object b) {
    var type = Types.typeof(a);
    if (!type.equals(Types.typeof(b))) return false;
    return type != Symbol.BOOLEAN || b == Boolean.TRUE;
  }

  public static Object left(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.head() == Symbol.EQUALS) return a1.get(1);
    }
    return a;
  }

  public static Object right(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.head() == Symbol.EQUALS) return a1.get(2);
    }
    return true;
  }
}
