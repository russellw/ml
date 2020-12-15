package specs;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;
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
    assertEquals(Code.eval(map, Array.of(Op.LT, 1, 1)), false);
    assertEquals(Code.eval(map, Array.of(Op.LT, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Op.LT, 2, 1)), false);
    assertEquals(Code.eval(map, Array.of(Op.LE, 1, 1)), true);
    assertEquals(Code.eval(map, Array.of(Op.LE, 1, 2)), true);
    assertEquals(Code.eval(map, Array.of(Op.LE, 2, 1)), false);
  }
}
