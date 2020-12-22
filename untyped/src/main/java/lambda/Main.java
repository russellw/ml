package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.Seq;
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

  public static void main(String[] args) {
    var s = Code.exprs(0);
    System.out.println(s);
    System.exit(0);
    var specs = new ArrayList<>();
    var tries = 0;
    while (specs.size() < 20) {
      tries++;
      try {
        var a = Code.exprs(1);
        var b = (Seq) Code.eval(a);
        if (!(b.get(2) instanceof Seq)) continue;
        // Code.simplify(HashMap.empty(), Array.of(Symbol.CALL, b, 0));
        specs.add(b);
        Code.println(a);
        Code.println(b);
        System.out.println();
      } catch (ArithmeticException
          | NoSuchElementException
          | UnsupportedOperationException ignored) {
      }
    }
    System.out.println(tries);
    Object best = null;
    var bestScore = -1;
    for (var i = 0; i < 1000000000; i++) {
      var a = Code.exprs(1);
      var score = test(specs, a);
      if (score > bestScore) {
        Code.println(a);
        System.out.println(score);
        best = a;
        bestScore = score;
      }
    }
  }
}
