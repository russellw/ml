package prover;

import io.vavr.collection.Seq;

public final class Types {
  public static Object typeof(Object a) {
    if (a instanceof Seq) a = ((Seq) a).head();
    if (a instanceof Func) return ((Func) a).isBoolean ? Symbol.BOOLEAN : Symbol.INDIVIDUAL;
    if (a instanceof Variable) return ((Variable) a).type;
    if (a instanceof Symbol)
      switch ((Symbol) a) {
        case EQUALS:
          return Symbol.BOOLEAN;
      }
    if (a instanceof Boolean) return Symbol.BOOLEAN;
    throw new IllegalArgumentException(a.toString());
  }
}
