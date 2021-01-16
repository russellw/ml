package prover;

import java.math.BigInteger;
import java.nio.file.Path;
import java.util.*;
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

  public static <T> List<List<T>> cartesianProduct(List<List<T>> qs) {
    var js = new int[qs.size()];
    var rs = new ArrayList<List<T>>();
    cartesianProduct(qs, 0, js, rs);
    return rs;
  }

  private static <T> void cartesianProduct(List<List<T>> qs, int i, int[] js, List<List<T>> rs) {
    if (i == js.length) {
      var ys = new ArrayList<T>();
      for (i = 0; i < js.length; i++) ys.add(qs.get(i).get(js[i]));
      rs.add(ys);
      return;
    }
    for (js[i] = 0; js[i] < qs.get(i).size(); js[i]++) cartesianProduct(qs, i + 1, js, rs);
  }

  public static String baseName(String file) {
    return withoutExtension(withoutDir(file));
  }

  public static void debug(Object a) {
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(a);
  }

  public static String withoutExtension(String file) {
    return file.split("\\.")[0];
  }

  public static String withoutDir(String file) {
    return Path.of(file).getFileName().toString();
  }

  public static <T> int count(List<T> a, Predicate<T> f) {
    var n = 0;
    for (var b : a) if (f.test(b)) n++;
    return n;
  }

  public static List<Object> map(List<Object> a, Function<Object, Object> f) {
    var r = new ArrayList<>();
    for (var b : a) r.add(f.apply(b));
    return r;
  }

  public static Object[] removeAt(Object[] a, int i) {
    var r = new Object[a.length - 1];
    System.arraycopy(a, 0, r, 0, i);
    System.arraycopy(a, i + 1, r, i, r.length - i);
    return r;
  }

  @SuppressWarnings("unchecked")
  public static Object mapLeaves(Object a, Function<Object, Object> f) {
    if (a instanceof List) {
      var a1 = (List) a;
      return map(a1, b -> mapLeaves(b, f));
    }
    return f.apply(a);
  }

  public static void walkLeaves(Object a, Consumer<Object> f) {
    if (a instanceof List) {
      var a1 = (List) a;
      for (var b : a1) walkLeaves(b, f);
      return;
    }
    f.accept(a);
  }

  public static Object splice(Object a, List<Integer> position, int i, Object b) {
    if (i == position.size()) return b;
    var a1 = (List) a;
    var r = new Object[a1.size()];
    for (var j = 0; j < r.length; j++) {
      var x = a1.get(j);
      if (j == position.get(i)) x = splice(x, position, i + 1, b);
      r[j] = x;
    }
    return List.of(r);
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

  public static boolean exists(List<Object> a, Predicate<Object> f) {
    for (var b : a) if (f.test(b)) return true;
    return false;
  }

  public static boolean all(List<Object> a, Predicate<Object> f) {
    for (var b : a) if (!f.test(b)) return false;
    return true;
  }

  public static boolean existsLeaf(Object a, Predicate<Object> f) {
    if (a instanceof List) {
      var a1 = (List) a;
      for (var b : a1) if (existsLeaf(b, f)) return true;
      return false;
    }
    return (f.test(a));
  }

  public static Set<Object> collectLeaves(Object a, Predicate<Object> f) {
    var r = new HashSet<>();
    collectLeaves(a, f, r);
    return r;
  }

  public static void collectLeaves(Object a, Predicate<Object> f, Set<Object> r) {
    if (a instanceof List) {
      var a1 = (List) a;
      for (var b : a1) collectLeaves(b, f, r);
      return;
    }
    if (f.test(a)) r.add(a);
  }

  public static Object head(Object a) {
    if (a instanceof List) return ((List) a).get(0);
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
