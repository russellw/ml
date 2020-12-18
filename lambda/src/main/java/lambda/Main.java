package lambda;

import io.vavr.collection.List;
import io.vavr.collection.Seq;
import java.util.NoSuchElementException;

public class Main {
  public static void main(String[] args) {
    var types = new Object[] {Symbol.BOOL, Symbol.INT};
    for (var type : types)
      for (var i = 0; i < 1000; i++)
        try {
          var a = Code.rand(List.empty(), type, 4);
          if (!(a instanceof Seq)) continue;
          var b = Code.simplify(List.empty(), a);
          System.out.println(a);
          System.out.println(b);
          System.out.println();
        } catch (ArithmeticException
            | GaveUp
            | NoSuchElementException
            | UnsupportedOperationException ignored) {
        }
  }
}
