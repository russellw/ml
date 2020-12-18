package lambda;

import io.vavr.collection.List;
import io.vavr.collection.Seq;

public class Main {
  public static void main(String[] args) {
    var types = new Object[] {Symbol.BOOL, Symbol.INT};
    var env = List.empty();
    for (var type : types)
      for (var i = 0; i < 1000; i++)
        try {
          var a = Code.rand(env, type, 4);
          if (!(a instanceof Seq)) continue;
          System.out.println(a);
          a = Code.eval(env, a);
          System.out.println(a);
          System.out.println();
        } catch (GaveUp ignored) {
        }
  }
}
