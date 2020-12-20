package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import io.vavr.collection.Seq;
import java.util.NoSuchElementException;

public class Main {
  public static void main(String[] args) {
    for (var i = 0; i < 100; i++)
      try {
        var a = Code.rand(List.empty(), Array.of(Symbol.FUNCTION, Symbol.INT, Symbol.BOOL), 4);
        var b = (Seq) Code.simplify(List.empty(), a);
        if (!(b.get(2) instanceof Seq)) continue;
        Code.println(a);
        Code.println(b);
        System.out.println();
      } catch (ArithmeticException
          | GaveUp
          | NoSuchElementException
          | UnsupportedOperationException ignored) {
      }
  }
}
