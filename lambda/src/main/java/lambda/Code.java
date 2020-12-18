package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;

public final class Code {
  private static Random random = new Random(0);

  public static Object rand(Seq env, Object type, int depth) {
    // random.nextInt() % n where n is a power of 2, avoids a divide instruction
    if (depth == 0 || random.nextInt() % 4 == 0) {
      var leaves = new ArrayList<>();
      if (accepts(type, Symbol.BOOL)) {
        leaves.add(false);
        leaves.add(true);
      }
      if (accepts(type, Symbol.INT)) {
        leaves.add(0);
        leaves.add(1);
      }
      if (accepts(type, Symbol.LIST)) {
        leaves.add(List.empty());
      }
      for (var symbol : Symbol.values()) if (accepts(type, typeof(env, symbol))) leaves.add(symbol);
      // despite being lists rather than atoms, argument references count as leaves
      // because they do not contain subexpressions; the index is a constant
      var i = 0;
      for (var argType : env) {
        if (accepts(type, argType)) leaves.add(Array.of(Symbol.ARG, i));
        i++;
      }
      if (leaves.isEmpty()) throw new GaveUp(type.toString());
      return leaves.get(random.nextInt(leaves.size()));
    }

    // compound expression
    depth--;

    // special forms
    switch (random.nextInt() % 16) {
      case 0:
        if (!accepts(type, Symbol.BOOL)) break;
        return Array.of(Symbol.AND, rand(env, Symbol.BOOL, depth), rand(env, Symbol.BOOL, depth));
      case 1:
        {
          if (!accepts(type, Symbol.BOOL)) break;
          var a = rand(env, Symbol.OBJECT, depth);
          var b = rand(env, typeof(env, a), depth);
          return Array.of(Symbol.EQ, a, b);
        }
      case 2:
        {
          var test = rand(env, Symbol.BOOL, depth);
          var a = rand(env, type, depth);
          var b = rand(env, typeof(env, a), depth);
          return Array.of(Symbol.IF, test, a, b);
        }
      case 3:
        if (!accepts(type, Symbol.BOOL)) break;
        return Array.of(Symbol.OR, rand(env, Symbol.BOOL, depth), rand(env, Symbol.BOOL, depth));
    }

    // function call
    var f = rand(env, Array.of(Symbol.FUNCTION, Symbol.OBJECT, type), depth);
    var functionType = (Seq) typeof(env, f);
    assert functionType.head() == Symbol.FUNCTION;
    var a = rand(env, functionType.get(1), depth);
    return Array.of(f, a);
  }

  private static void accept(Object paramType, Object argType) {
    if (!accepts(paramType, argType)) throw new TypeError(paramType + " != " + argType);
  }

  private static boolean accepts(Object paramType, Object argType) {
    if (paramType == argType) return true;
    if (paramType == Symbol.OBJECT) return true;
    if (paramType instanceof Seq) {
      var paramType1 = (Seq) paramType;
      if (argType instanceof Seq) {
        var argType1 = (Seq) argType;
        var n = paramType1.size();
        if (n != argType1.size()) return false;
        for (var i = 0; i < n; i++) if (!accepts(paramType1.get(i), argType1.get(i))) return false;
        return true;
      }
    }
    return false;
  }

  private static Object combine(Object t, Object u) {
    if (t == u) return t;
    if (t == Symbol.OBJECT) return u;
    if (u == Symbol.OBJECT) return t;
    if (t instanceof Seq) {
      var t1 = (Seq) t;
      if (u instanceof Seq) {
        var u1 = (Seq) u;
        var n = t1.size();
        if (n != u1.size()) throw new TypeError(t + " != " + u);
        var r = new Object[n];
        for (var i = 0; i < n; i++) r[i] = combine(t1.get(i), u1.get(i));
        return Array.of(r);
      }
    }
    throw new TypeError(t + " != " + u);
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
            accept(Symbol.BOOL, typeof(env, a1.get(1)));
            accept(Symbol.BOOL, typeof(env, a1.get(2)));
            return Symbol.BOOL;
          case EQ:
            combine(typeof(env, a1.get(1)), typeof(env, a1.get(2)));
            return Symbol.BOOL;
          case IF:
            accept(Symbol.BOOL, typeof(env, a1.get(1)));
            return combine(typeof(env, a1.get(2)), typeof(env, a1.get(3)));
          case LAMBDA:
            {
              var paramType = a1.get(1);
              var body = a1.get(2);
              return Array.of(Symbol.FUNCTION, paramType, typeof(env.prepend(paramType), body));
            }
          case ARG:
            return env.get((int) a1.get(1));
        }
      var functionType = (Seq) typeof(env, a1.head());
      if (functionType.head() != Symbol.FUNCTION) throw new TypeError(a.toString());
      accept(functionType.get(1), typeof(env, a1.get(1)));
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
    return Symbol.SYMBOL;
  }

  @SuppressWarnings("unchecked")
  public static Object simplify(Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    if (a1.isEmpty()) return a;
    var o = a1.head();
    if (o instanceof Symbol)
      switch ((Symbol) o) {
        case LAMBDA:
        case ARG:
          return a;
      }
    a1 = a1.map(Code::simplify);
    if (a1.forAll(Code::constant)) return eval(List.empty(), a1);
    o = a1.head();
    if (o instanceof Symbol)
      switch ((Symbol) o) {
        case AND:
          {
            var x = a1.get(1);
            var y = a1.get(2);
            if (x == Boolean.FALSE || y == Boolean.FALSE) return false;
            if (x == Boolean.TRUE) return y;
            if (y == Boolean.TRUE) return x;
            break;
          }
        case OR:
          {
            var x = a1.get(1);
            var y = a1.get(2);
            if (x == Boolean.TRUE || y == Boolean.TRUE) return true;
            if (x == Boolean.FALSE) return y;
            if (y == Boolean.FALSE) return x;
            break;
          }
        case IF:
          {
            var test = a1.get(1);
            var x = a1.get(2);
            var y = a1.get(3);
            if (x.equals(y)) return x;
            if (test == Boolean.TRUE) return x;
            if (test == Boolean.FALSE) return y;
            break;
          }
        case EQ:
          {
            var x = a1.get(1);
            var y = a1.get(2);
            if (x.equals(y)) return true;
            break;
          }
      }
    return a1;
  }

  private static boolean constant(Object a) {
    if (a instanceof Seq) return ((Seq) a).isEmpty();
    return true;
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
