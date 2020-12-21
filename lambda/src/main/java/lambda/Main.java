package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.NoSuchElementException;

public class Main {
  public static void main(String[] args) {
    var specs = new ArrayList<>();
    var i = 0;
    while (specs.size() < 20) {
      i++;
      try {
        var a = Code.rand(List.empty(), Array.of(Symbol.FUNCTION, Symbol.OBJECT, Symbol.OBJECT), 5);
        var b = (Seq) Code.simplify(HashMap.empty(), a);
        if (!(b.get(2) instanceof Seq)) continue;
        Code.simplify(HashMap.empty(), Array.of(Symbol.CALL, b, 0));
        specs.add(b);
        Code.println(a);
        Code.println(b);
        System.out.println();
      } catch (ArithmeticException
          | GaveUp
          | NoSuchElementException
          | UnsupportedOperationException ignored) {
      }
    }
    System.out.println(i);
  }
}
