package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Random;

public final class Code {
  private static Variable X = new Variable(null);
  private static Variable Y = new Variable(null);
  private static Pattern[] patterns =
      new Pattern[] {
        new Pattern(Symbol.ADD, 0, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.ADD, X, 0) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.SUB, X, X) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.SUB, X, 0) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.MUL, 0, X) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.MUL, X, 0) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.MUL, 1, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.MUL, X, 1) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.DIV, 0, X) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.DIV, X, 1) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.DIV, X, X) {
          @Override
          Object output() {
            return 1;
          }
        },
        new Pattern(Symbol.REM, 0, X) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.REM, X, 1) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.REM, X, X) {
          @Override
          Object output() {
            return 0;
          }
        },
        new Pattern(Symbol.NOT, false) {
          @Override
          Object output() {
            return true;
          }
        },
        new Pattern(Symbol.NOT, true) {
          @Override
          Object output() {
            return false;
          }
        },
        new Pattern(Symbol.AND, X, false) {
          @Override
          Object output() {
            return false;
          }
        },
        new Pattern(Symbol.AND, false, X) {
          @Override
          Object output() {
            return false;
          }
        },
        new Pattern(Symbol.AND, X, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.AND, true, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.AND, X, true) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.OR, X, false) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.OR, false, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.OR, X, X) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.OR, true, X) {
          @Override
          Object output() {
            return true;
          }
        },
        new Pattern(Symbol.OR, X, true) {
          @Override
          Object output() {
            return true;
          }
        },
        new Pattern(Symbol.EQ, X, X) {
          @Override
          Object output() {
            return true;
          }
        },
        new Pattern(Symbol.LT, X, X) {
          @Override
          Object output() {
            return false;
          }
        },
        new Pattern(Symbol.LE, X, X) {
          @Override
          Object output() {
            return true;
          }
        },
        new Pattern(Symbol.IF, X, Y, Y) {
          @Override
          Object output() {
            return Y;
          }
        },
        new Pattern(Symbol.IF, true, X, Y) {
          @Override
          Object output() {
            return X;
          }
        },
        new Pattern(Symbol.IF, false, X, Y) {
          @Override
          Object output() {
            return Y;
          }
        },
      };
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
        leaves.add(quote(List.empty()));
      }
      // Despite being lists rather than atoms, argument references count as leaves
      // because they do not contain subexpressions; the index is a constant
      var i = 0;
      for (var argType : env) {
        if (accepts(type, argType)) leaves.add(Array.of(Symbol.ARG, i));
        i++;
      }
      if (!leaves.isEmpty()) return leaves.get(random.nextInt(leaves.size()));
      if (depth == 0) throw new GaveUp(type.toString());
    }
    depth--;
    for (var i = 0; i < 1000; i++) {
      var symbols = Symbol.values();
      var o = symbols[random.nextInt(symbols.length)];
      switch (o) {
        case CALL:
          {
            var f = rand(env, Array.of(Symbol.FUNCTION, Symbol.OBJECT, type), depth);
            var functionType = (Seq) typeof(env, f);
            assert functionType.head() == Symbol.FUNCTION;
            var a = rand(env, functionType.get(1), depth);
            return Array.of(o, f, a);
          }
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
      switch ((Symbol) a1.head()) {
        case CALL:
          {
            var functionType = (Seq) typeof(env, a1.get(1));
            if (functionType.head() != Symbol.FUNCTION) throw new TypeError(a.toString());
            var paramType = functionType.get(1);
            var argType = typeof(env, a1.get(2));
            accept(paramType, argType);
            return functionType.get(2);
          }
        case QUOTE:
          {
            var x = a1.get(1);
            if (x instanceof Seq) return Symbol.LIST;
            if (x instanceof Boolean) return Symbol.BOOL;
            if (x instanceof Integer) return Symbol.INT;
            if (x instanceof Symbol) return Symbol.SYMBOL;
            throw new IllegalArgumentException(a.toString());
          }
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
          typeof(env, a1.get(1));
          typeof(env, a1.get(2));
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
    }
    if (a instanceof Boolean) return Symbol.BOOL;
    if (a instanceof Integer) return Symbol.INT;
    if (a instanceof Symbol) return Symbol.SYMBOL;
    throw new IllegalArgumentException(a.toString());
  }

  public static void println(Object a) {
    print(a);
    System.out.println();
  }

  public static void print(Object a) {
    if (!(a instanceof Seq)) {
      System.out.print(a);
      return;
    }
    var a1 = (Seq) a;
    if (a1.head() == Symbol.ARG) {
      print(a1.head());
      System.out.print(a1.get(1));
      return;
    }
    System.out.print('(');
    for (var i = 0; i < a1.size(); i++) {
      if (i > 0) System.out.print(' ');
      print(a1.get(i));
    }
    System.out.print(')');
  }

  private static Object unquote(Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    if (a1.head() == Symbol.QUOTE) return a1.get(1);
    return null;
  }

  public static Object quote(Object a) {
    if (a instanceof Seq) return Array.of(Symbol.QUOTE, a);
    return a;
  }

  @SuppressWarnings("unchecked")
  public static Object simplify(Seq<Variable> env, Object a) {
    // Atom or compound?
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    var o = (Symbol) a1.head();

    // Special forms
    switch (o) {
      case ARG:
        {
          var i = (int) a1.get(1);
          var value = env.get(i).value;
          if (value == null) return a;
          return value;
        }
      case LAMBDA:
        {
          var paramType = a1.get(1);
          var body = simplify(env.prepend(new Variable(paramType)), a1.get(2));
          return Array.of(o, paramType, body);
        }
      case QUOTE:
        return quote(a1.get(1));
      case CALL:
        {
          var x = (Seq) a1.get(1);
          var y = a1.get(2);
          if (x.head() == Symbol.LAMBDA) {
            var y1 = unquote(y);
            if (y1 != null) {
              var paramType = x.get(1);
              var body = x.get(2);
              // return simplify(env.prepend(new Variable(paramType, quote(y1))), body);
            }
          }
          break;
        }
    }

    // Simplify subterms
    a = a1.map(b -> simplify(env, b));

    // Patterns
    for (; ; ) {
      var old = a;
      for (var p : patterns) a = p.transform(a);
      if (a.equals(old)) break;
    }

    // Atom or compound?
    if (!(a instanceof Seq)) return a;
    a1 = (Seq) a;
    o = (Symbol) a1.head();

    // Evaluate if possible
    var x = unquote(a1.get(1));
    if (x == null) return a;
    Object y = null;
    if (a1.size() > 2) {
      y = unquote(a1.get(2));
      if (y == null) return a;
    }
    switch (o) {
      case HEAD:
        return quote(((Seq) x).head());
      case TAIL:
        return quote(((Seq) x).tail());
      case CONS:
        assert y != null;
        return quote(((Seq) y).prepend(x));
      case LT:
        return (int) x < (int) y;
      case LE:
        return (int) x <= (int) y;
      case ADD:
        return (int) x + (int) y;
      case SUB:
        return (int) x - (int) y;
      case DIV:
        return (int) x / (int) y;
      case REM:
        return (int) x % (int) y;
      case MUL:
        return (int) x * (int) y;
      case EQ:
        // X=X evaluates to true
        // Therefore to have got this far:
        // Arguments must be syntactically unequal
        // Arguments must be constants
        // Therefore they must be actually unequal
        return false;
    }
    return a;
  }

  @SuppressWarnings("unchecked")
  public static Object replace(Object a, Map<Variable, Object> map) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      return a1.map(b -> replace(b, map));
    }
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      var a2 = map.getOrElse(a1, null);
      if (a2 != null) return replace(a2, map);
    }
    return a;
  }

  public static Map<Variable, Object> match(Object a, Object b, Map<Variable, Object> map) {
    if (a == b) return map;
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      var a2 = map.getOrElse(a1, null);
      if (a2 == null) return map.put(a1, b);
      if (a2.equals(b)) return map;
      return null;
    }
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (b instanceof Seq) {
        var b1 = (Seq) b;
        var n = a1.size();
        if (n != b1.size()) return null;
        for (var i = 0; i < n; i++) {
          map = match(a1.get(i), b1.get(i), map);
          if (map == null) return null;
        }
        return map;
      }
    }
    if (a.equals(b)) return map;
    return null;
  }
}
