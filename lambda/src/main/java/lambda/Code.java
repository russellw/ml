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
      if (type == Symbol.BOOL || type == Symbol.OBJECT) {
        leaves.add(false);
        leaves.add(true);
      }
      if (type == Symbol.INT || type == Symbol.OBJECT) {
        leaves.add(0);
        leaves.add(1);
      }
      if (type == Symbol.LIST || type == Symbol.OBJECT) {
        leaves.add(List.empty());
      }
      for (var a : Symbol.values()) if (combine(typeof(env, a), type) != null) leaves.add(a);
      // despite being lists rather than atoms, argument references count as leaves
      // because they do not contain subexpressions; the index is a constant
      var i = 0;
      for (var argType : env) {
        if (combine(argType, type) != null) leaves.add(Array.of(Symbol.ARG, i));
        i++;
      }
      if (leaves.isEmpty()) throw new IllegalArgumentException(type.toString());
      return leaves.get(random.nextInt(leaves.size()));
    }

    // compound expression
    depth--;

    // special forms
    switch (random.nextInt() % 8) {
      case 0:
        if (type != Symbol.BOOL) break;
        return Array.of(Symbol.AND, rand(env, Symbol.BOOL, depth), rand(env, Symbol.BOOL, depth));
      case 1:
        {
          if (type != Symbol.BOOL) break;
          var a = rand(env, Symbol.OBJECT, depth);
          var b = rand(env, typeof(env, a), depth);
          return Array.of(Symbol.EQ, a, b);
        }
      case 2:
        {
          var test = rand(env, Symbol.BOOL, depth);
          var a = rand(env, Symbol.OBJECT, depth);
          var b = rand(env, typeof(env, a), depth);
          return Array.of(Symbol.IF, test, a, b);
        }
      case 3:
        if (type != Symbol.BOOL) break;
        return Array.of(Symbol.OR, rand(env, Symbol.BOOL, depth), rand(env, Symbol.BOOL, depth));
    }

    // function call
    var f = rand(env, Array.of(Symbol.FUNCTION, Symbol.OBJECT, type), depth);
    var functionType = typeof(env, f);
    Object argType;
    if (functionType == Symbol.OBJECT) argType = Symbol.OBJECT;
    else {
      if (!(functionType instanceof Seq))
        throw new IllegalStateException(type.toString() + ": " + f + ": " + functionType);
      var functionType1 = (Seq) functionType;
      if (functionType1.head() != Symbol.FUNCTION)
        throw new IllegalStateException(type.toString() + ": " + f + ": " + functionType);
      argType = functionType1.get(1);
    }
    var a = rand(env, argType, depth);
    return Array.of(f, a);
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
            combine(typeof(env, a1.get(1)), Symbol.BOOL);
            combine(typeof(env, a1.get(2)), Symbol.BOOL);
            return Symbol.BOOL;
          case EQ:
            combine(typeof(env, a1.get(1)), typeof(env, a1.get(2)));
            return Symbol.BOOL;
          case IF:
            {
              combine(typeof(env, a1.get(1)), Symbol.BOOL);
              var type = typeof(env, a1.get(2));
              combine(type, typeof(env, a1.get(3)));
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
      var functionType = typeof(env, a1.head());
      if (functionType == Symbol.OBJECT) return Symbol.OBJECT;
      var functionType1 = (Seq) functionType;
      if (functionType1.head() != Symbol.FUNCTION) throw new TypeError(a.toString());
      combine(typeof(env, a1.get(1)), functionType1.get(1));
      return functionType1.get(2);
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
