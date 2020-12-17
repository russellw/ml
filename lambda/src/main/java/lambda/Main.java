package lambda;

import io.vavr.collection.List;

public class Main {
  public static void main(String[] args) {
    for (var i = 0; i < 10; i++) {
      var a = Code.rand(List.empty(), Symbol.BOOL, 3);
      System.out.println(a);
    }
  }
}
