package specs;

import io.vavr.collection.*;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

public final class Code {
  private static Random random = new Random();

  private static Object combine(Object type1, Object type2) {
    if (Objects.equals(type1, type2)) return type1;
    if (type1 == Symbol.OBJECT) return type2;
    if (type2 == Symbol.OBJECT) return type1;
    return null;
  }

  public static Object typeof(Object a) {
    if (a instanceof Boolean) return Symbol.BOOL;
    if (a instanceof Integer) return Symbol.INT;
    if (a instanceof Symbol)
      switch ((Symbol) a) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case REM:
          return Array.of(Symbol.INT, Symbol.INT, Symbol.INT);
        case AND:
        case OR:
          return Array.of(Symbol.BOOL, Symbol.BOOL, Symbol.BOOL);
        case NOT:
          return Array.of(Symbol.BOOL, Symbol.BOOL);
        case LE:
        case LT:
          return Array.of(Symbol.BOOL, Symbol.INT, Symbol.INT);
        case EQ:
          return Array.of(Symbol.BOOL, Symbol.OBJECT, Symbol.OBJECT);
        case HEAD:
          return Array.of(Symbol.OBJECT, Symbol.LIST);
        case TAIL:
          return Array.of(Symbol.LIST, Symbol.LIST);
        case CONS:
          return Array.of(Symbol.LIST, Symbol.OBJECT, Symbol.LIST);
      }
    var a1 = (Seq) a;
    if (a1.isEmpty()) return Symbol.LIST;
    var ftype = (Seq) typeof(a1.head());
    if (ftype == null) return null;
    if (a1.size() != ftype.size()) return null;
    for (var i = 1; i < a1.size(); i++) {
      var type = combine(typeof(a1.get(i)), ftype.get(i));
      if (type == null) return null;
    }
    return ftype.head();
  }

  private static int arity(Symbol op) {
    switch (op) {
      case NOT:
      case HEAD:
      case TAIL:
        return 1;
    }
    return 2;
  }

  public static Seq<Object> leaves() {
    return Array.of(0, 1, List.empty());
  }

  private static boolean constant(Object a) {
    if (a instanceof Boolean) return true;
    if (a instanceof Integer) return true;
    if (a instanceof Symbol) return true;
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      for (var b : a1) if (!constant(b)) return false;
      return true;
    }
    return false;
  }

  public static Object simplify(Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    for (var b : a1) if (!constant(b)) return a;
    return eval(HashMap.empty(), a);
  }

  public static Object rand(Seq<Object> leaves, int depth) {
    if (depth == 0 || random.nextInt(5) == 0) return leaves.get(random.nextInt(leaves.size()));
    var symbols = Symbol.values();
    var op = symbols[random.nextInt(symbols.length)];
    var n = arity(op);
    var r = new ArrayList<>(n + 1);
    r.add(op);
    for (var i = 0; i < n; i++) r.add(rand(leaves, depth - 1));
    return Array.ofAll(r);
  }

  public static Object eval(Map<Object, Object> map, Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    if (a1.isEmpty()) return a;
    var op = (Symbol) a1.head();
    switch (op) {
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
        return eval(map, a1.get(1)).equals(eval(map, a1.get(2)));
      case LT:
        return (int) eval(map, a1.get(1)) < (int) eval(map, a1.get(2));
      case LE:
        return (int) eval(map, a1.get(1)) <= (int) eval(map, a1.get(2));
      case AND:
        return (boolean) eval(map, a1.get(1)) && (boolean) eval(map, a1.get(2));
      case OR:
        return (boolean) eval(map, a1.get(1)) || (boolean) eval(map, a1.get(2));
      case NOT:
        return !(boolean) eval(map, a1.get(1));
      case HEAD:
        return ((Seq) eval(map, a1.get(1))).head();
      case TAIL:
        return ((Seq) eval(map, a1.get(1))).tail();
      case CONS:
        {
          var x = eval(map, a1.get(1));
          var s = (Seq) eval(map, a1.get(2));
          @SuppressWarnings("unchecked")
          var r = s.prepend(x);
          return r;
        }
    }
    throw new IllegalArgumentException(a.toString());
  }
}
