package lambda;

import io.vavr.collection.Array;
import java.util.ArrayList;
import java.util.NoSuchElementException;

public class Main {
  private static boolean test(Object spec, Object a) {
    try {
      var value = Code.eval(Array.of(a, spec));
      var r = Code.eval(Array.of(spec, value));
      return (boolean) r;
    } catch (ArithmeticException | NoSuchElementException | UnsupportedOperationException ignored) {
      return false;
    }
  }

  private static int test(ArrayList<Object> specs, Object a) {
    var n = 0;
    for (var spec : specs) if (test(spec, a)) n++;
    return n;
  }

  private static boolean isIntPredicate(Object a) {
    try {
      Code.eval(Array.of(a, 0));
      return true;
    } catch (ArithmeticException
        | ClassCastException
        | IndexOutOfBoundsException
        | NoSuchElementException
        | UnsupportedOperationException ignored) {
      return false;
    }
  }

  public static void main(String[] args) {
    var s = new ArrayList<>();
    for (var i = 0; i < 1000; i++) {
      var a = Code.rand(5);
      if (isIntPredicate(a)) s.add(a);
    }
    for (var a : s) Code.println(a);
    System.out.println(s.size());
  }
}
