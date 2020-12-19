package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import java.util.NoSuchElementException;

public class Main {
  public static void main(String[] args) {
    for (var i = 0; i < 1000; i++)
      try {
        var a = Code.rand(List.empty(), Array.of(Symbol.FUNCTION, Symbol.INT, Symbol.BOOL), 4);
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
