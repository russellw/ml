package lambda;

import io.vavr.collection.List;

public class Main {
  public static void main(String[] args) {
    for (var i = 0; i < 100; i++)
      try {
        var a = Code.rand(List.empty(), Symbol.BOOL, 4);
        System.out.println(a);
      } catch (GaveUp ignored) {
      }
  }
}
