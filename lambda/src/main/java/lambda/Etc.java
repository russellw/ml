package lambda;

public final class Etc {
  public static void debug(Object a) {
    System.err.print(Thread.currentThread().getStackTrace()[2] + ": ");
    System.err.println(a);
  }
}
