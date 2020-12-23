package lambda;

import io.vavr.collection.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;

public final class Code {
  private static Random random = new Random(0);

  public static Object eval(Object a) {
    return eval(HashMap.empty(), a);
  }

  private static Integer toInt(boolean b) {
    return b ? 1 : 0;
  }

  private static boolean toBoolean(Object a) {
    return (int) a != 0;
  }

  @SuppressWarnings("unchecked")
  private static Object eval(Map<Variable, Object> map, Object a) {
    if (a instanceof Variable) return map.get((Variable) a).get();
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    if (a1.isEmpty()) return a;
    var o = a1.head();
    if (o instanceof Symbol)
      switch ((Symbol) o) {
        case ADD:
          return (int) eval(map, a1.get(1)) + (int) eval(map, a1.get(2));
        case SUB:
          return (int) eval(map, a1.get(1)) - (int) eval(map, a1.get(2));
        case MUL:
          return (int) eval(map, a1.get(1)) * (int) eval(map, a1.get(2));
        case DIV:
          return (int) eval(map, a1.get(1)) / (int) eval(map, a1.get(2));
        case REM:
          return (int) eval(map, a1.get(1)) % (int) eval(map, a1.get(2));
        case EQ:
          return toInt(eval(map, a1.get(1)).equals(eval(map, a1.get(2))));
        case LT:
          return toInt((int) eval(map, a1.get(1)) < (int) eval(map, a1.get(2)));
        case LE:
          return toInt((int) eval(map, a1.get(1)) <= (int) eval(map, a1.get(2)));
        case AND:
          return toInt(toBoolean(eval(map, a1.get(1))) && toBoolean(eval(map, a1.get(2))));
        case OR:
          return toInt(toBoolean(eval(map, a1.get(1))) || toBoolean(eval(map, a1.get(2))));
        case NOT:
          return toInt(!toBoolean(eval(map, a1.get(1))));
        case HEAD:
          return ((Seq) eval(map, a1.get(1))).head();
        case TAIL:
          return ((Seq) eval(map, a1.get(1))).tail();
        case LAMBDA:
          {
            var param = (Variable) a1.get(1);
            var body = a1.get(2);
            return (Function) b -> eval(map.put(param, b), body);
          }
        case CONS:
          {
            var x = eval(map, a1.get(1));
            var s = (Seq) eval(map, a1.get(2));
            return s.prepend(x);
          }
      }
    o = eval(map, o);
    var f = (Function) o;
    var b = eval(map, a1.get(1));
    return f.apply(b);
  }

  public static ArrayList<Object> terms(int depth, Predicate<Object> select) {
    return terms(depth, select, List.empty());
  }

  private static void add(Predicate<Object> select, Object a, ArrayList<Object> r) {
    if (select == null || select.test(a)) r.add(a);
  }

  private static ArrayList<Object> terms(
      int depth, Predicate<Object> select, Seq<Variable> variables) {
    var r = new ArrayList<>();
    if (depth == 0) {
      add(select, 0, r);
      add(select, 1, r);
      add(select, List.empty(), r);
      for (var x : variables) add(select, x, r);
      return r;
    }
    for (var i = 0; i < depth; i++) r.addAll(terms(i, null, variables));
    depth--;
    for (var o : Symbol.values())
      switch (o) {
        case HEAD:
        case TAIL:
        case NOT:
          {
            var xs = terms(depth, null, variables);
            for (var x : xs) add(select, Array.of(o, x), r);
            break;
          }
        case LAMBDA:
          {
            var param = new Variable();
            var xs = terms(depth, null, variables.prepend(param));
            for (var x : xs) add(select, Array.of(o, param, x), r);
            break;
          }
        case IF:
          {
            var xs = terms(depth, null, variables);
            var ys = terms(depth, null, variables);
            var zs = terms(depth, null, variables);
            for (var x : xs) for (var y : ys) for (var z : zs) add(select, Array.of(o, x, y, z), r);
            break;
          }
        default:
          {
            var xs = terms(depth, null, variables);
            var ys = terms(depth, null, variables);
            for (var x : xs) for (var y : ys) add(select, Array.of(o, x, y), r);
            break;
          }
      }
    return r;
  }

  public static void println(Object a) {
    print(new java.util.HashMap<>(), a);
    System.out.println();
  }

  private static String variableName(int i) {
    if (i < 26) return Character.toString('A' + i);
    return "Z" + (i - 25);
  }

  private static void print(java.util.Map<Variable, String> map, Object a) {
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
    System.out.print('(');
    for (var i = 0; i < a1.size(); i++) {
      if (i > 0) System.out.print(' ');
      print(map, a1.get(i));
    }
    System.out.print(')');
  }
}
