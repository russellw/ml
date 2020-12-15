package specs;

import io.vavr.collection.Array;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Random;

public final class Code {
  private static Random random = new Random();

  private static int arity(Op op) {
    return 2;
  }

  public static Seq<Object> leaves() {
    return Array.of(0, 1);
  }

  private static boolean constant(Object a) {
    if (a instanceof Boolean) return true;
    if (a instanceof Integer) return true;
    return false;
  }

  public static Object simplify(Object a) {
    return a;
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
    var op = (Op) a1.get(0);
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
    }
    throw new IllegalArgumentException(a.toString());
  }
}
