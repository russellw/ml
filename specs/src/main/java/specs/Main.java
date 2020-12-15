package specs;

import io.vavr.collection.HashMap;

public class Main {
  public static void main(String[] args) {
    var map = HashMap.empty();
    for (var i = 0; i < 10; i++) {
      var a = Code.rand(Code.leaves(), 3);
      System.out.println(a);
      try {
        System.out.println(Code.eval(map, a));
      } catch (ArithmeticException | ClassCastException e) {
        System.out.println(e.toString());
      }
      System.out.println();
    }
  }
}
