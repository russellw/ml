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
    return a instanceof Integer;
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
    try {
      switch (op) {
        case ADD:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x + y;
          }
        case SUB:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x - y;
          }
        case MUL:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x * y;
          }
        case DIV:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x / y;
          }
        case REM:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x % y;
          }
        case EQ:
          {
            var x = eval(map, a1.get(1));
            var y = eval(map, a1.get(2));
            return x.equals(y);
          }
      }
    } catch (ArithmeticException e) {
      return 0;
    }
    throw new IllegalArgumentException(a.toString());
  }
}
