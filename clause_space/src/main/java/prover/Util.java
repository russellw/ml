package prover;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;

public final class Util {
  private static int depth;

  public static void debug(Object a) {
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(a);
  }

  public static void debug(Object[] q) {
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(Arrays.asList(q));
  }

  public static void debugIn(Object a) {
    indent(depth++);
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(a);
  }

  public static void debugOut(Object a) {
    indent(--depth);
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(a);
  }

  public static void debugTime() {
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(
        LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")));
  }

  private static void indent(int depth) {
    for (var i = 0; i < depth; i++) {
      System.err.print(' ');
    }
  }
}
