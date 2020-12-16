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
    assertEquals(Code.eval(map, Array.of(Op.ADD, 1, 2)), 3);
    assertEquals(Code.eval(map, Array.of(Op.SUB, 1, 2)), -1);
    assertEquals(Code.eval(map, Array.of(Op.MUL, 2, 3)), 6);
    assertEquals(Code.eval(map, Array.of(Op.DIV, 10, 3)), 3);
    assertEquals(Code.eval(map, Array.of(Op.REM, 10, 3)), 1);
    assertEquals(Code.eval(map, Array.of(Op.EQ, 10, 10)), true);
    assertEquals(Code.eval(map, Array.of(Op.EQ, 10, 11)), false);
    assertEquals(Code.eval(map, Array.of(Op.EQ, List.empty(), Array.empty())), true);
    assertEquals(Code.eval(map, Array.of(Op.LT, 1, 1)), false);
    assertEquals(Code.eval(map, Array.of(Op.LT, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Op.LT, 2, 1)), false);
    assertEquals(Code.eval(map, Array.of(Op.LE, 1, 1)), true);
    assertEquals(Code.eval(map, Array.of(Op.LE, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Op.LE, 2, 1)), false);
    assertEquals(Code.eval(map, Array.of(Op.AND, false, false)), false);
    assertEquals(Code.eval(map, Array.of(Op.AND, false, true)), false);
    assertEquals(Code.eval(map, Array.of(Op.AND, true, false)), false);
    assertEquals(Code.eval(map, Array.of(Op.AND, true, true)), true);
    assertEquals(Code.eval(map, Array.of(Op.OR, false, false)), false);
    assertEquals(Code.eval(map, Array.of(Op.OR, false, true)), true);
    assertEquals(Code.eval(map, Array.of(Op.OR, true, false)), true);
    assertEquals(Code.eval(map, Array.of(Op.OR, true, true)), true);
    assertEquals(Code.eval(map, Array.of(Op.NOT, false)), true);
    assertEquals(Code.eval(map, Array.of(Op.NOT, true)), false);
    assertEquals(Code.eval(map, Array.of(Op.CONS, 1, List.empty())), Array.of(1));
    assertEquals(Code.eval(map, Array.of(Op.CONS, 1, Array.empty())), List.of(1));
    assertEquals(Code.eval(map, Array.of(Op.HEAD, Array.of(Op.CONS, 1, Array.empty()))), 1);
    assertEquals(
        Code.eval(map, Array.of(Op.TAIL, Array.of(Op.CONS, 1, Array.empty()))), List.empty());
  }

  @Test
  public void typeof() {
    assertEquals(Code.typeof(1), BasicType.INT);
    assertEquals(Code.typeof(true), BasicType.BOOL);
    assertEquals(Code.typeof(Array.empty()), BasicType.LIST);
    assertEquals(Code.typeof(Array.of(Op.ADD, 1, 2)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Op.SUB, 1, 2)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Op.MUL, 2, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Op.DIV, 10, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Op.REM, 10, 3)), BasicType.INT);
    assertEquals(Code.typeof(Array.of(Op.EQ, 10, 10)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.LT, 1, 1)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.LE, 1, 1)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.AND, false, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.OR, false, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.NOT, false)), BasicType.BOOL);
    assertEquals(Code.typeof(Array.of(Op.CONS, 1, List.empty())), BasicType.LIST);
    assertEquals(
        Code.typeof(Array.of(Op.HEAD, Array.of(Op.CONS, 1, Array.empty()))), BasicType.OBJECT);
    assertEquals(
        Code.typeof(Array.of(Op.TAIL, Array.of(Op.CONS, 1, Array.empty()))), BasicType.LIST);
  }
}
