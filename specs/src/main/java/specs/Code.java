package specs;

import io.vavr.collection.*;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

public final class Code {
  private static Random random = new Random();

  private static Object combine(Object t, Object u) {
    if (Objects.equals(t, u)) return t;
    if (t == BasicType.OBJECT) return u;
    if (u == BasicType.OBJECT) return t;
    return null;
  }

  public static Object typeof(Object a) {
    if (a instanceof Boolean) return BasicType.BOOL;
    if (a instanceof Integer) return BasicType.INT;
    if (a instanceof Op)
      switch ((Op) a) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case REM:
          return Array.of(BasicType.INT, BasicType.INT, BasicType.INT);
        case AND:
        case OR:
          return Array.of(BasicType.BOOL, BasicType.BOOL, BasicType.BOOL);
        case NOT:
          return Array.of(BasicType.BOOL, BasicType.BOOL);
        case LE:
        case LT:
          return Array.of(BasicType.BOOL, BasicType.INT, BasicType.INT);
        case EQ:
          return Array.of(BasicType.BOOL, BasicType.OBJECT, BasicType.OBJECT);
        case HEAD:
          return Array.of(BasicType.OBJECT, BasicType.LIST);
        case TAIL:
          return Array.of(BasicType.LIST, BasicType.LIST);
        case CONS:
          return Array.of(BasicType.LIST, BasicType.OBJECT, BasicType.LIST);
      }
    var a1 = (Seq) a;
    if (a1.isEmpty()) return BasicType.LIST;
    var ft = (Seq) typeof(a1.head());
    if (ft == null) return null;
    if (a1.size() != ft.size()) return null;
    for (var i = 1; i < a1.size(); i++) {
      var t = combine(typeof(a1.get(i)), ft.get(i));
      if (t == null) return null;
    }
    return ft.head();
  }

  private static int arity(Op op) {
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
    if (a instanceof Op) return true;
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
    var ops = Op.values();
    var op = ops[random.nextInt(ops.length)];
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
    var op = (Op) a1.head();
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
