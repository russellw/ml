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
  }
}
