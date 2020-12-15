package specs;

import io.vavr.collection.Array;
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
}
