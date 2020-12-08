package prover;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

public final class Util {
  public static final Path stdin = Path.of("stdin");
  private static int depth;

  private Util() {}

  public static <T> List<List<T>> cartesianProduct(List<List<T>> qs) {
    var js = new int[qs.size()];
    var rs = new ArrayList<List<T>>();
    cartesianProduct(qs, 0, js, rs);
    return rs;
  }

  private static <T> void cartesianProduct(List<List<T>> qs, int i, int[] js, List<List<T>> rs) {
    if (i == js.length) {
      var ys = new ArrayList<T>();
      for (i = 0; i < js.length; i++) {
        ys.add(qs.get(i).get(js[i]));
      }
      rs.add(ys);
      return;
    }
    for (js[i] = 0; js[i] < qs.get(i).size(); js[i]++) {
      cartesianProduct(qs, i + 1, js, rs);
    }
  }

  public static <T> int count(Iterable<T> q, Predicate<T> f) {
    var n = 0;
    for (var a : q) {
      if (f.test(a)) {
        n++;
      }
    }
    return n;
  }

  public static void debug(Object a) {
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(a);
  }

  public static void debug(Object[] q) {
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(Arrays.asList(q));
  }

  public static void debugIn(Object a) {
    indent(depth++);
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(a);
  }

  public static void debugOut(Object a) {
    indent(--depth);
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(a);
  }

  public static void debugTime() {
    System.out.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.out.println(
        LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")));
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

  public static void endElement(XMLStreamWriter writer, int depth) throws XMLStreamException {
    indent(writer, depth);
    writer.writeEndElement();
  }

  public static String extension(String file) {
    var i = file.lastIndexOf('.');
    if (i < 0) {
      return "";
    }
    return file.substring(i + 1);
  }

  private static void indent(int depth) {
    for (var i = 0; i < depth; i++) {
      System.out.print(' ');
    }
  }

  public static void indent(XMLStreamWriter writer, int depth) throws XMLStreamException {
    for (var i = 0; i < depth; i++) {
      writer.writeCharacters("  ");
    }
  }

  public static boolean isDigits(String s) {
    if (s.isEmpty()) {
      return false;
    }
    for (int i = 0; i < s.length(); i++) {
      if (!Character.isDigit(s.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  public static boolean isInteger(double a) {
    return a == Math.floor(a);
  }

  public static void printFile(String path) throws IOException {
    System.out.println(Files.readString(Path.of(path), StandardCharsets.UTF_8));
  }

  public static String quote(char q, String s) {
    var sb = new StringBuilder();
    sb.append(q);
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if ((c == q) || (c == '\\')) {
        sb.append('\\');
      }
      sb.append(c);
    }
    sb.append(q);
    return sb.toString();
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

  public static <T> List<T> remove(List<T> q, int i) {
    var r = new ArrayList<T>(q.size() - 1);
    for (var j = 0; j < q.size(); j++) {
      if (j != i) {
        r.add(q.get(j));
      }
    }
    return r;
  }

  public static String removeExtension(String file) {
    var i = file.lastIndexOf('.');
    if (i < 0) {
      return file;
    }
    return file.substring(0, i);
  }

  public static <T> List<T> replace(List<T> q, int i, T a) {
    var r = new ArrayList<T>(q.size());
    for (var j = 0; j < q.size(); j++) {
      r.add((j == i) ? a : q.get(j));
    }
    return r;
  }

  public static void startElement(XMLStreamWriter writer, int depth, String name)
      throws XMLStreamException {
    indent(writer, depth);
    writer.writeStartElement(name);
  }

  public static void text(XMLStreamWriter writer, int depth, String s) throws XMLStreamException {
    indent(writer, depth);
    writer.writeCharacters(s);
  }

  public static double truncate(double a) {
    return (a < 0) ? Math.ceil(a) : Math.floor(a);
  }
}
