package lambda;

import io.vavr.collection.*;
import java.util.ArrayList;
import java.util.Random;

public final class Code {
  private static Variable B = new Variable(null);
  private static Variable X = new Variable(null);
  private static Variable Y = new Variable(null);
  private static Pattern[] patterns =
      new Pattern[] {
        new Pattern(Symbol.CALL, Array.of(Symbol.LAMBDA, X, B), Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var body = map.get(B).get();
            var x = (Variable) map.get(X).get();
            var y = map.get(Y).get();
            var body1 = simplify(map, replace(body, HashMap.of(x, y)));
            if (size(body1) > size(body) + 4) return null;
            return body1;
          }
        },
        new Pattern(Symbol.ADD, 0, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.ADD, X, 0) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.EQ, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = unquote(map.get(X).get());
            if (x == null) return null;
            var y = unquote(map.get(Y).get());
            if (y == null) return null;
            return x.equals(y);
          }
        },
        new Pattern(Symbol.CONS, X, Y) {
          @Override
          @SuppressWarnings("unchecked")
          Object output(Map<Variable, Object> map) {
            var x = unquote(map.get(X).get());
            if (x == null) return null;
            var y = unquote(map.get(Y).get());
            if (y == null) return null;
            return quote(((Seq) y).prepend(x));
          }
        },
        new Pattern(Symbol.HEAD, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = unquote(map.get(X).get());
            if (x == null) return null;
            return quote(((Seq) x).head());
          }
        },
        new Pattern(Symbol.TAIL, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = unquote(map.get(X).get());
            if (x == null) return null;
            return quote(((Seq) x).tail());
          }
        },
        new Pattern(Symbol.ADD, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 + y1;
          }
        },
        new Pattern(Symbol.SUB, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 - y1;
          }
        },
        new Pattern(Symbol.MUL, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 * y1;
          }
        },
        new Pattern(Symbol.DIV, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 / y1;
          }
        },
        new Pattern(Symbol.REM, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 % y1;
          }
        },
        new Pattern(Symbol.LT, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 < y1;
          }
        },
        new Pattern(Symbol.LE, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            var x = map.get(X).get();
            if (!(x instanceof Integer)) return null;
            var x1 = (int) x;
            var y = map.get(Y).get();
            if (!(y instanceof Integer)) return null;
            var y1 = (int) y;
            return x1 <= y1;
          }
        },
        new Pattern(Symbol.SUB, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.SUB, X, 0) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.MUL, 0, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.MUL, X, 0) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.MUL, 1, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.MUL, X, 1) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.DIV, 0, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.DIV, X, 1) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.DIV, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 1;
          }
        },
        new Pattern(Symbol.REM, 0, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.REM, X, 1) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.REM, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return 0;
          }
        },
        new Pattern(Symbol.NOT, false) {
          @Override
          Object output(Map<Variable, Object> map) {
            return true;
          }
        },
        new Pattern(Symbol.NOT, true) {
          @Override
          Object output(Map<Variable, Object> map) {
            return false;
          }
        },
        new Pattern(Symbol.AND, X, false) {
          @Override
          Object output(Map<Variable, Object> map) {
            return false;
          }
        },
        new Pattern(Symbol.AND, false, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return false;
          }
        },
        new Pattern(Symbol.AND, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.AND, true, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.AND, X, true) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.OR, X, false) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.OR, false, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.OR, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.OR, true, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return true;
          }
        },
        new Pattern(Symbol.OR, X, true) {
          @Override
          Object output(Map<Variable, Object> map) {
            return true;
          }
        },
        new Pattern(Symbol.EQ, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return true;
          }
        },
        new Pattern(Symbol.LT, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return false;
          }
        },
        new Pattern(Symbol.LE, X, X) {
          @Override
          Object output(Map<Variable, Object> map) {
            return true;
          }
        },
        new Pattern(Symbol.IF, X, Y, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            return Y;
          }
        },
        new Pattern(Symbol.IF, true, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            return X;
          }
        },
        new Pattern(Symbol.IF, false, X, Y) {
          @Override
          Object output(Map<Variable, Object> map) {
            return Y;
          }
        },
      };
  private static Random random = new Random(0);

  public static Object rand(Seq<Variable> env, Object type, int depth) {
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
      for (var x : env) if (accepts(type, x.type)) leaves.add(x);
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
            var functionType = (Seq) typeof(f);
            assert functionType.head() == Symbol.FUNCTION;
            var a = rand(env, functionType.get(1), depth);
            return Array.of(o, f, a);
          }
        case LAMBDA:
          {
            if (!(type instanceof Seq)) break;
            var type1 = (Seq) type;
            if (type1.head() != Symbol.FUNCTION) break;
            var param = new Variable(type1.get(1));
            var returnType = type1.get(2);
            var body = rand(env.prepend(param), returnType, depth);
            return Array.of(o, param, body);
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
            var b = rand(env, typeof(a), depth);
            return Array.of(o, a, b);
          }
        case IF:
          {
            var test = rand(env, Symbol.BOOL, depth);
            var a = rand(env, type, depth);
            var b = rand(env, typeof(a), depth);
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

  public static Object typeof(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      switch ((Symbol) a1.head()) {
        case CALL:
          {
            var functionType = (Seq) typeof(a1.get(1));
            if (functionType.head() != Symbol.FUNCTION) throw new TypeError(a.toString());
            var paramType = functionType.get(1);
            var argType = typeof(a1.get(2));
            accept(paramType, argType);
            return functionType.get(2);
          }
        case QUOTE:
          a = a1.get(1);
          if (a instanceof Seq) return Symbol.LIST;
          break;
        case HEAD:
          accept(Symbol.LIST, typeof(a1.get(1)));
          return Symbol.OBJECT;
        case TAIL:
          accept(Symbol.LIST, typeof(a1.get(1)));
          return Symbol.LIST;
        case CONS:
          accept(Symbol.OBJECT, typeof(a1.get(1)));
          accept(Symbol.LIST, typeof(a1.get(2)));
          return Symbol.LIST;
        case LE:
        case LT:
          accept(Symbol.INT, typeof(a1.get(1)));
          accept(Symbol.INT, typeof(a1.get(2)));
          return Symbol.BOOL;
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case REM:
          accept(Symbol.INT, typeof(a1.get(1)));
          accept(Symbol.INT, typeof(a1.get(2)));
          return Symbol.INT;
        case NOT:
          accept(Symbol.BOOL, typeof(a1.get(1)));
          return Symbol.BOOL;
        case AND:
        case OR:
          accept(Symbol.BOOL, typeof(a1.get(1)));
          accept(Symbol.BOOL, typeof(a1.get(2)));
          return Symbol.BOOL;
        case EQ:
          typeof(a1.get(1));
          typeof(a1.get(2));
          return Symbol.BOOL;
        case IF:
          accept(Symbol.BOOL, typeof(a1.get(1)));
          return combine(typeof(a1.get(2)), typeof(a1.get(3)));
        case LAMBDA:
          {
            var param = (Variable) a1.get(1);
            var body = a1.get(2);
            return Array.of(Symbol.FUNCTION, param.type, typeof(body));
          }
      }
    }
    if (a instanceof Boolean) return Symbol.BOOL;
    if (a instanceof Integer) return Symbol.INT;
    if (a instanceof Symbol) return Symbol.SYMBOL;
    if (a instanceof Variable) return ((Variable) a).type;
    throw new IllegalArgumentException(a.toString());
  }

  public static void println(Object a) {
    print(a, new java.util.HashMap<>());
    System.out.println();
  }

  private static String variableName(int i) {
    if (i < 26) return Character.toString('A' + i);
    return "Z" + (i - 25);
  }

  private static void print(Object a, java.util.Map<Variable, String> map) {
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      var name = map.get(a1);
      if (name == null) {
        name = variableName(map.size());
        map.put(a1, name);
      }
      System.out.print(name);
      return;
    }
    if (!(a instanceof Seq)) {
      System.out.print(a);
      return;
    }
    var a1 = (Seq) a;
    if (!a1.isEmpty())
      switch ((Symbol) a1.head()) {
        case LAMBDA:
          System.out.print('{');
          var param = (Variable) a1.get(1);
          print(param, map);
          System.out.print(':');
          print(param.type, map);
          System.out.print(' ');
          print(a1.get(2), map);
          System.out.print('}');
          return;
        case QUOTE:
          System.out.print('\'');
          print(a1.get(1), map);
          return;
      }
    System.out.print('(');
    for (var i = 0; i < a1.size(); i++) {
      if (i > 0) System.out.print(' ');
      print(a1.get(i), map);
    }
    System.out.print(')');
  }

  private static Object unquote(Object a) {
    if (a instanceof Variable) return null;
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
  public static Object simplify(Map<Variable, Object> env, Object a) {
    simplify:
    for (; ; ) {
      // Variable
      if (a instanceof Variable) {
        var a1 = (Variable) a;
        var r = env.getOrElse(a1, null);
        if (r == null) return a;
        return r;
      }

      // Constant
      if (!(a instanceof Seq)) return a;

      // Compound
      var a1 = (Seq) a;
      var o = (Symbol) a1.head();

      // Special syntax
      if (o == Symbol.QUOTE) return quote(a1.get(1));

      // Simplify subterms
      a = a1.map(b -> simplify(env, b));

      // Patterns
      for (var p : patterns) {
        var b = p.transform(a, env);
        if (b != null) {
          a = b;
          continue simplify;
        }
      }
      return a;
    }
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

  private static int size(Object a) {
    if (!(a instanceof Seq)) return 1;
    var a1 = (Seq) a;
    var n = 0;
    for (var b : a1) n += size(b);
    return n;
  }
}
