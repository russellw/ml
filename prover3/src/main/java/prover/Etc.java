package prover;

import io.vavr.collection.*;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

public final class Etc {
  private Etc() {}

  public static String extension(String file) {
    var i = file.lastIndexOf('.');
    if (i < 0) return "";
    return file.substring(i + 1);
  }

  public static <T> ArrayList<List<T>> cartesianProduct(ArrayList<List<T>> qs) {
    var js = new int[qs.size()];
    var rs = new ArrayList<List<T>>();
    cartesianProduct(qs, 0, js, rs);
    return rs;
  }

  private static <T> void cartesianProduct(
      ArrayList<List<T>> qs, int i, int[] js, ArrayList<List<T>> rs) {
    if (i == js.length) {
      var ys = new ArrayList<T>();
      for (i = 0; i < js.length; i++) ys.add(qs.get(i).get(js[i]));
      rs.add(ys);
      return;
    }
    for (js[i] = 0; js[i] < qs.get(i).size(); js[i]++) cartesianProduct(qs, i + 1, js, rs);
  }

  public static Seq<Object> implies(Object a, Object b) {
    return Array.of(Symbol.OR, Array.of(Symbol.NOT, a), b);
  }

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

  public static void treeWalk(Object a, Consumer<Object> f) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      for (var b : a1) treeWalk(b, f);
      return;
    }
    f.accept(a);
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
    if (a.signum() < 0 != b.signum() < 0 && r[1].signum() != 0)
      r[0] = r[0].subtract(BigInteger.ONE);
    return r[0];
  }

  public static BigInteger remainderEuclidean(BigInteger a, BigInteger b) {
    var r = a.remainder(b);
    if (r.signum() < 0) r = r.add(b.abs());
    return r;
  }

  public static BigInteger remainderFloor(BigInteger a, BigInteger b) {
    return a.subtract(divideFloor(a, b).multiply(b));
  }

  public static boolean treeExists(Object a, Predicate<Object> f) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      for (var b : a1) if (treeExists(b, f)) return true;
      return false;
    }
    return (f.test(a));
  }

  public static java.util.HashSet<Object> collect(Object a, Predicate<Object> f) {
    var r = new java.util.HashSet<>();
    collect(a, f, r);
    return r;
  }

  public static void collect(Object a, Predicate<Object> f, java.util.HashSet<Object> r) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      for (var b : a1) collect(b, f, r);
      return;
    }
    if (f.test(a)) r.add(a);
  }

  public static Object head(Object a) {
    if (a instanceof Seq) return ((Seq) a).head();
    return null;
  }

  public static String quote(char q, String s) {
    var sb = new StringBuilder();
    sb.append(q);
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if (c == q || c == '\\') sb.append('\\');
      sb.append(c);
    }
    sb.append(q);
    return sb.toString();
  }
}
