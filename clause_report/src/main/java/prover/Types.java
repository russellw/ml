package prover;

import io.vavr.collection.Seq;
import java.math.BigInteger;

public final class Types {
  private Types() {}

  public static Object typeof(Object a) {
    if (a instanceof Seq) a = ((Seq) a).head();
    if (a instanceof Func) return ((Func) a).type;
    if (a instanceof Variable) return ((Variable) a).type;
    if (a instanceof Symbol)
      switch ((Symbol) a) {
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
      }
    if (a instanceof Boolean) return Symbol.BOOLEAN;
    if (a instanceof BigInteger) return Symbol.INTEGER;
    if (a instanceof BigRational) return Symbol.RATIONAL;
    if (a instanceof String) return Symbol.INDIVIDUAL;
    throw new IllegalArgumentException(a + ": " + a.getClass());
  }
}
