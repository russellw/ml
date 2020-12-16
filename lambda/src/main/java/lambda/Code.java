package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

public final class Code {
  private static Random random = new Random();

  public static Object rand(Seq env, Object type, int depth) {
    if (depth == 0 || random.nextInt() % 4 == 0) {
      var leaves = new ArrayList<>();
      if (type instanceof Symbol)
        switch ((Symbol) type) {
          case BOOL:
            leaves.add(false);
            leaves.add(true);
            break;
          case INT:
            leaves.add(0);
            leaves.add(1);
            break;
        }
      for (var a : Symbol.values()) if (combine(typeof(env, a), type) != null) leaves.add(a);
      return leaves.get(random.nextInt(leaves.size()));
    }
    throw new IllegalArgumentException();
  }

  private static Object combine(Object type1, Object type2) {
    if (Objects.equals(type1, type2)) return type1;
    if (type1 == Symbol.OBJECT) return type2;
    if (type2 == Symbol.OBJECT) return type1;
    return null;
  }

  @SuppressWarnings("unchecked")
  public static Object typeof(Seq env, Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.isEmpty()) return Symbol.LIST;
      var o = a1.head();
      if (o instanceof Symbol)
        switch ((Symbol) o) {
          case AND:
          case OR:
            if (typeof(env, a1.get(1)) != Symbol.BOOL) return null;
            if (typeof(env, a1.get(2)) != Symbol.BOOL) return null;
            return Symbol.BOOL;
          case EQ:
            if (combine(typeof(env, a1.get(1)), typeof(env, a1.get(2))) == null) return null;
            return Symbol.BOOL;
          case IF:
            {
              if (typeof(env, a1.get(1)) != Symbol.BOOL) return null;
              var type = typeof(env, a1.get(2));
              if (combine(type, typeof(env, a1.get(3))) == null) return null;
              return type;
            }
          case LAMBDA:
            {
              var param = a1.get(1);
              var body = a1.get(2);
              return Array.of(Symbol.FUNCTION, param, typeof(env.prepend(param), body));
            }
          case ARG:
            return env.get((int) a1.get(1));
        }
      var functionType = (Seq) typeof(env, a1.head());
      if (functionType == null) return null;
      if (functionType.head() != Symbol.FUNCTION) return null;
      if (combine(typeof(env, a1.get(1)), functionType.get(1)) == null) return null;
      return functionType.get(2);
    }
    if (a instanceof Symbol)
      switch ((Symbol) a) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case REM:
          return Array.of(
              Symbol.FUNCTION, Symbol.INT, Array.of(Symbol.FUNCTION, Symbol.INT, Symbol.INT));
        case NOT:
          return Array.of(Symbol.FUNCTION, Symbol.BOOL, Symbol.BOOL);
        case LE:
        case LT:
          return Array.of(
              Symbol.FUNCTION, Symbol.INT, Array.of(Symbol.FUNCTION, Symbol.INT, Symbol.BOOL));
        case HEAD:
          return Array.of(Symbol.FUNCTION, Symbol.LIST, Symbol.OBJECT);
        case TAIL:
          return Array.of(Symbol.FUNCTION, Symbol.LIST, Symbol.LIST);
        case CONS:
          return Array.of(
              Symbol.FUNCTION, Symbol.OBJECT, Array.of(Symbol.FUNCTION, Symbol.LIST, Symbol.LIST));
      }
    if (a instanceof Integer) return Symbol.INT;
    if (a instanceof Boolean) return Symbol.BOOL;
    throw new IllegalArgumentException(a.toString());
  }

  @SuppressWarnings("unchecked")
  public static Object eval(Seq env, Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.isEmpty()) return a;
      var o = a1.head();
      if (o instanceof Symbol)
        switch ((Symbol) o) {
          case IF:
            return (boolean) eval(env, a1.get(1)) ? eval(env, a1.get(2)) : eval(env, a1.get(3));
          case AND:
            return (boolean) eval(env, a1.get(1)) && (boolean) eval(env, a1.get(2));
          case OR:
            return (boolean) eval(env, a1.get(1)) || (boolean) eval(env, a1.get(2));
          case EQ:
            return eval(env, a1.get(1)).equals(eval(env, a1.get(2)));
          case LAMBDA:
            {
              var body = a1.get(2);
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
