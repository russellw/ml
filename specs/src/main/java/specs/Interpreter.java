package specs;

import io.vavr.collection.Map;
import io.vavr.collection.Seq;

public final class Interpreter {
  public Object eval(Map<Object, Object> map, Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    var op = (Op) a1.get(0);
    try {
      switch (op) {
        case ADD:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x + y;
          }
        case SUB:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x - y;
          }
        case MUL:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x * y;
          }
        case DIV:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x / y;
          }
        case REM:
          {
            var x = (int) eval(map, a1.get(1));
            var y = (int) eval(map, a1.get(2));
            return x % y;
          }
      }
    } catch (ArithmeticException e) {
      return 0;
    }
    throw new IllegalArgumentException(a.toString());
  }
}
