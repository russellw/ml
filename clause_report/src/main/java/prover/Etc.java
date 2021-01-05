package prover;

import io.vavr.collection.Array;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.function.Consumer;
import java.util.function.Function;

public final class Etc {
  private Etc() {}

  public static void debug(Object a) {
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(a);
  }

  @SuppressWarnings("unchecked")
  public static Object treeMap(Object a, Function<Object, Object> f) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      return a1.map(b -> treeMap(b, f));
    }
    return f.apply(a);
  }

  public static void treeForEach(Object a, Consumer<Object> f) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      for (var b : a1) treeForEach(b, f);
      return;
    }
    f.accept(a);
  }

  public static Object replace(Object a, Map<Variable, Object> map) {
    return treeMap(
        a,
        b -> {
          if (b instanceof Variable) {
            var b1 = map.getOrElse((Variable) b, null);
            if (b1 != null) return replace(b1, map);
          }
          return b;
        });
  }

  public static Object splice(Object a, ArrayList<Integer> position, int i, Object b) {
    if (i == position.size()) return b;
    var a1 = (Seq) a;
    var r = new Object[a1.size()];
    for (var j = 0; j < r.length; j++) {
      var x = a1.get(j);
      if (j == position.get(i)) x = splice(x, position, i + 1, b);
      r[j] = x;
    }
    return Array.of(r);
  }

  public static BigInteger divideEuclidean(BigInteger a, BigInteger b) {
    return a.subtract(remainderEuclidean(a, b)).divide(b);
  }

  public static BigInteger divideFloor(BigInteger a, BigInteger b) {
    var r = a.divideAndRemainder(b);
    if ((a.signum() < 0 != b.signum() < 0) && (r[1].signum() != 0)) {
      r[0] = r[0].subtract(BigInteger.ONE);
    }
    return r[0];
  }

  public static BigInteger remainderEuclidean(BigInteger a, BigInteger b) {
    var r = a.remainder(b);
    if (r.signum() < 0) {
      r = r.add(b.abs());
    }
    return r;
  }

  public static BigInteger remainderFloor(BigInteger a, BigInteger b) {
    return a.subtract(divideFloor(a, b).multiply(b));
  }
}
