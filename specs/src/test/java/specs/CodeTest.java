package specs;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import org.junit.Test;

public class CodeTest {
  @Test
  public void simplify() {
    assertEquals(Code.simplify(1), 1);
  }

  @Test
  public void eval() {
    var map = HashMap.empty();
    assertEquals(Code.eval(map, 0), 0);
    assertEquals(Code.eval(map, Array.of(Symbol.ADD, 1, 2)), 3);
    assertEquals(Code.eval(map, Array.of(Symbol.SUB, 1, 2)), -1);
    assertEquals(Code.eval(map, Array.of(Symbol.MUL, 2, 3)), 6);
    assertEquals(Code.eval(map, Array.of(Symbol.DIV, 10, 3)), 3);
    assertEquals(Code.eval(map, Array.of(Symbol.REM, 10, 3)), 1);
    assertEquals(Code.eval(map, Array.of(Symbol.EQ, 10, 10)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.EQ, 10, 11)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.EQ, List.empty(), Array.empty())), true);
    assertEquals(Code.eval(map, Array.of(Symbol.LT, 1, 1)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.LT, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.LT, 2, 1)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.LE, 1, 1)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.LE, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.LE, 2, 1)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.AND, false, false)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.AND, false, true)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.AND, true, false)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.AND, true, true)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.OR, false, false)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.OR, false, true)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.OR, true, false)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.OR, true, true)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.NOT, false)), true);
    assertEquals(Code.eval(map, Array.of(Symbol.NOT, true)), false);
    assertEquals(Code.eval(map, Array.of(Symbol.CONS, 1, List.empty())), Array.of(1));
    assertEquals(Code.eval(map, Array.of(Symbol.CONS, 1, Array.empty())), List.of(1));
    assertEquals(Code.eval(map, Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.empty()))), 1);
    assertEquals(
        Code.eval(map, Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.empty()))),
        List.empty());
  }

  @Test
  public void typeof() {
    assertEquals(Code.typeof(1), BasicType.INT);
    assertEquals(Code.typeof(true), BasicType.BOOL);
    assertEquals(Code.typeof(Array.empty()), BasicType.LIST);
    assertEquals(Code.typeof(Array.of(Symbol.ADD, 1, 2)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Symbol.SUB, 1, 2)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Symbol.MUL, 2, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Symbol.DIV, 10, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Symbol.REM, 10, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Symbol.EQ, 10, 10)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.LT, 1, 1)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.LE, 1, 1)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.AND, false, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.OR, false, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.NOT, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.CONS, 1, List.empty())), BasicType.LIST);
    assertEquals(
        Code.typeof(Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.empty()))),
        BasicType.OBJECT);
    assertEquals(
        Code.typeof(Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.empty()))),
        BasicType.LIST);
  }
}
