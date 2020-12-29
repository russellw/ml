package prover;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;

public final class Etc {
  private static int depth;

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

  private static void indent(int depth) {
    for (var i = 0; i < depth; i++) {
      System.out.print(' ');
    }
  }
}
