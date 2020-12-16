package lambda;

import io.vavr.collection.Seq;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

public final class Code {
  private static Random random = new Random();

  private static Object combine(Object type1, Object type2) {
    if (Objects.equals(type1, type2)) return type1;
    if (type1 == Symbol.OBJECT) return type2;
    if (type2 == Symbol.OBJECT) return type1;
    return null;
  }

  @SuppressWarnings("unchecked")
  public static Object eval(Seq env, Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.isEmpty()) return a;
      var o = a1.head();
      if (o instanceof Symbol)
        switch ((Symbol) o) {
          case AND:
            return (boolean) eval(env, a1.get(1)) && (boolean) eval(env, a1.get(2));
          case OR:
            return (boolean) eval(env, a1.get(1)) || (boolean) eval(env, a1.get(2));
          case LAMBDA:
            {
              var body = a1.get(1);
              return (Function) x -> eval(env.prepend(x), body);
            }
          case ARG:
            return env.get((int) a1.get(1));
        }
      o = eval(env, o);
      return ((Function) o).apply(eval(env, a1.get(1)));
    }
    if (a instanceof Symbol)
      switch ((Symbol) a) {
        case ADD:
          return (Function) x -> (Function) y -> (int) x + (int) y;
        case SUB:
          return (Function) x -> (Function) y -> (int) x - (int) y;
        case MUL:
          return (Function) x -> (Function) y -> (int) x * (int) y;
        case DIV:
          return (Function) x -> (Function) y -> (int) x / (int) y;
        case REM:
          return (Function) x -> (Function) y -> (int) x % (int) y;
        case LE:
          return (Function) x -> (Function) y -> (int) x <= (int) y;
        case LT:
          return (Function) x -> (Function) y -> (int) x < (int) y;
        case EQ:
          return (Function) x -> (Function) x::equals;
        case NOT:
          return (Function) x -> !(boolean) x;
        case HEAD:
          return (Function) x -> ((Seq) x).head();
        case TAIL:
          return (Function) x -> ((Seq) x).tail();
        case CONS:
          return (Function) x -> (Function) y -> ((Seq) y).prepend(x);
      }
    return a;
  }
}
