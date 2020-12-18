package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;

public final class Code {
  private static Random random = new Random(0);

  @SuppressWarnings("unchecked")
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

    // function call
    if (random.nextInt() % 2 == 0) {
      var f = rand(env, Array.of(Symbol.FUNCTION, Symbol.OBJECT, type), depth);
      var functionType = (Seq) typeof(env, f);
      assert functionType.head() == Symbol.FUNCTION;
      var a = rand(env, functionType.get(1), depth);
      return Array.of(f, a);
    }

    // special forms
    for (var i = 0; i < 1000; i++) {
      var symbols = Symbol.values();
      var o = symbols[random.nextInt(symbols.length)];
      switch (o) {
        case LAMBDA:
          {
            if (!(type instanceof Seq)) break;
            var type1 = (Seq) type;
            if (type1.head() != Symbol.FUNCTION) break;
            var paramType = type1.get(1);
            var returnType = type1.get(2);
            var body = rand(env.prepend(paramType), returnType, depth);
            return Array.of(o, paramType, body);
          }
        case NOT:
          if (!accepts(type, Symbol.BOOL)) break;
          return Array.of(o, rand(env, Symbol.BOOL, depth));
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case REM:
          if (!accepts(type, Symbol.INT)) break;
          return Array.of(o, rand(env, Symbol.INT, depth), rand(env, Symbol.INT, depth));
        case LE:
        case LT:
          if (!accepts(type, Symbol.BOOL)) break;
          return Array.of(o, rand(env, Symbol.INT, depth), rand(env, Symbol.INT, depth));
        case AND:
        case OR:
          if (!accepts(type, Symbol.BOOL)) break;
          return Array.of(o, rand(env, Symbol.BOOL, depth), rand(env, Symbol.BOOL, depth));
        case HEAD:
          if (!accepts(type, Symbol.OBJECT)) break;
          return Array.of(o, rand(env, Symbol.LIST, depth));
        case TAIL:
          if (!accepts(type, Symbol.LIST)) break;
          return Array.of(o, rand(env, Symbol.LIST, depth));
        case CONS:
          if (!accepts(type, Symbol.LIST)) break;
          return Array.of(o, rand(env, Symbol.OBJECT, depth), rand(env, Symbol.LIST, depth));
        case EQ:
          {
            if (!accepts(type, Symbol.BOOL)) break;
            var a = rand(env, Symbol.OBJECT, depth);
            var b = rand(env, typeof(env, a), depth);
            return Array.of(o, a, b);
          }
        case IF:
          {
            var test = rand(env, Symbol.BOOL, depth);
            var a = rand(env, type, depth);
            var b = rand(env, typeof(env, a), depth);
            return Array.of(o, test, a, b);
          }
      }
    }
    throw new GaveUp(type.toString());
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
          case HEAD:
            accept(Symbol.LIST, typeof(env, a1.get(1)));
            return Symbol.OBJECT;
          case TAIL:
            accept(Symbol.LIST, typeof(env, a1.get(1)));
            return Symbol.LIST;
          case CONS:
            accept(Symbol.OBJECT, typeof(env, a1.get(1)));
            accept(Symbol.LIST, typeof(env, a1.get(2)));
            return Symbol.LIST;
          case LE:
          case LT:
            accept(Symbol.INT, typeof(env, a1.get(1)));
            accept(Symbol.INT, typeof(env, a1.get(2)));
            return Symbol.BOOL;
          case ADD:
          case SUB:
          case MUL:
          case DIV:
          case REM:
            accept(Symbol.INT, typeof(env, a1.get(1)));
            accept(Symbol.INT, typeof(env, a1.get(2)));
            return Symbol.INT;
          case NOT:
            accept(Symbol.BOOL, typeof(env, a1.get(1)));
            return Symbol.BOOL;
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
              var returnType = typeof(env.prepend(paramType), body);
              return Array.of(Symbol.FUNCTION, paramType, returnType);
            }
          case ARG:
            return env.get((int) a1.get(1));
        }
      var functionType = (Seq) typeof(env, a1.head());
      if (functionType.head() != Symbol.FUNCTION) throw new TypeError(a.toString());
      var paramType = functionType.get(1);
      var argType = typeof(env, a1.get(1));
      accept(paramType, argType);
      return functionType.get(2);
    }
    if (a instanceof Boolean) return Symbol.BOOL;
    if (a instanceof Integer) return Symbol.INT;
    if (a instanceof Symbol) return Symbol.SYMBOL;
    throw new IllegalArgumentException(a.toString());
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
            // if (x.equals(y)) return true;
            // if(constant(x)&&constant(y))return false;
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
    if (!(a instanceof Seq)) return a;
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
        case ADD:
          return (int) eval(env, a1.get(1)) + (int) eval(env, a1.get(2));
        case SUB:
          return (int) eval(env, a1.get(1)) - (int) eval(env, a1.get(2));
        case MUL:
          return (int) eval(env, a1.get(1)) * (int) eval(env, a1.get(2));
        case DIV:
          return (int) eval(env, a1.get(1)) / (int) eval(env, a1.get(2));
        case REM:
          return (int) eval(env, a1.get(1)) % (int) eval(env, a1.get(2));
        case LE:
          return (int) eval(env, a1.get(1)) <= (int) eval(env, a1.get(2));
        case LT:
          return (int) eval(env, a1.get(1)) < (int) eval(env, a1.get(2));
        case NOT:
          return !(boolean) eval(env, a1.get(1));
        case HEAD:
          return ((Seq) eval(env, a1.get(1))).head();
        case TAIL:
          return ((Seq) eval(env, a1.get(1))).tail();
        case CONS:
          {
            var x = eval(env, a1.get(1));
            var y = (Seq) eval(env, a1.get(2));
            return y.prepend(x);
          }
      }
    o = eval(env, o);
    return ((Function) o).apply(eval(env, a1.get(1)));
  }
}
