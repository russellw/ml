package specs;

import io.vavr.collection.HashMap;

public class Main {
  public static void main(String[] args) {
    var it = new Interpreter();
    var map = HashMap.empty();
    for (var i = 0; i < 10; i++) {
      var a = Code.rand(Code.leaves(), 3);
      System.out.println(a);
      System.out.println(it.eval(map, a));
      System.out.println();
    }
  }
}
