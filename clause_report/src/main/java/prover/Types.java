package prover;

import io.vavr.collection.Seq;
import java.math.BigInteger;

public final class Types {
  private Types() {}

  public static boolean isNumeric(Object a) {
    var t = typeof(a);
    if (t instanceof Symbol)
      switch ((Symbol) t) {
        case INTEGER:
        case RATIONAL:
        case REAL:
          return true;
      }
    return false;
  }

  public static Object typeof(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case EQUALS:
          case AND:
          case OR:
          case EQV:
          case NOT:
          case IS_INTEGER:
          case IS_RATIONAL:
          case EXISTS:
          case ALL:
          case LESS:
          case LESS_EQ:
            return Symbol.BOOLEAN;
          case TO_INTEGER:
            return Symbol.INTEGER;
          case TO_REAL:
            return Symbol.REAL;
          case TO_RATIONAL:
            return Symbol.RATIONAL;
          default:
            throw new IllegalArgumentException(a.getClass().toString() + ' ' + a);
        }
      var t = typeof(op);
      if (!(t instanceof Seq))
        throw new IllegalArgumentException(a.getClass().toString() + ' ' + a + ", " + t);
      var t1 = (Seq) t;
      if (t1.size() != a1.size())
        throw new IllegalArgumentException(a.getClass().toString() + ' ' + a);
      for (var i = 1; i < t1.size(); i++)
        if (!t1.get(i).equals(typeof(a1.get(i))))
          throw new IllegalArgumentException(a.getClass().toString() + ' ' + a);
      return t1.head();
    }
    if (a instanceof Func) return ((Func) a).type;
    if (a instanceof Variable) return ((Variable) a).type;
    if (a instanceof Boolean) return Symbol.BOOLEAN;
    if (a instanceof BigInteger) return Symbol.INTEGER;
    if (a instanceof BigRational) return Symbol.RATIONAL;
    if (a instanceof String) return Symbol.INDIVIDUAL;
    throw new IllegalArgumentException(a.getClass().toString() + ' ' + a);
  }
}
