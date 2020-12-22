package lambda;

import static org.junit.Assert.*;

import io.vavr.collection.*;
import org.junit.Test;

public class CodeTest {
  @Test
  public void eval() {
    assertEquals(Code.eval(1), 1);
    assertEquals(Code.eval(Array.of(Symbol.ADD, 1, 2)), 3);
    assertEquals(Code.eval(Array.of(Symbol.SUB, 1, (Object) 2)), -1);
    assertEquals(Code.eval(Array.of(Symbol.MUL, 2, (Object) 3)), 6);
    assertEquals(Code.eval(Array.of(Symbol.DIV, 10, (Object) 3)), 3);
    assertEquals(Code.eval(Array.of(Symbol.REM, 10, (Object) 3)), 1);
    assertEquals(Code.eval(Array.of(Symbol.EQ, 10, 10)), 1);
    assertEquals(Code.eval(Array.of(Symbol.EQ, 10, 11)), 0);
    assertEquals(Code.eval(Array.of(Symbol.LT, 1, (Object) 1)), 0);
    assertEquals(Code.eval(Array.of(Symbol.LT, 1, (Object) 2)), 1);
    assertEquals(Code.eval(Array.of(Symbol.LT, 2, (Object) 1)), 0);
    assertEquals(Code.eval(Array.of(Symbol.LE, 1, (Object) 1)), 1);
    assertEquals(Code.eval(Array.of(Symbol.LE, 1, (Object) 2)), 1);
    assertEquals(Code.eval(Array.of(Symbol.LE, 2, (Object) 1)), 0);
    assertEquals(Code.eval(Array.of(Symbol.AND, 0, 0)), 0);
    assertEquals(Code.eval(Array.of(Symbol.AND, 0, 1)), 0);
    assertEquals(Code.eval(Array.of(Symbol.AND, 1, 0)), 0);
    assertEquals(Code.eval(Array.of(Symbol.AND, 1, 1)), 1);
    assertEquals(Code.eval(Array.of(Symbol.OR, 0, 0)), 0);
    assertEquals(Code.eval(Array.of(Symbol.OR, 0, 1)), 1);
    assertEquals(Code.eval(Array.of(Symbol.OR, 1, 0)), 1);
    assertEquals(Code.eval(Array.of(Symbol.OR, 1, 1)), 1);
    assertEquals(Code.eval(Array.of(Symbol.NOT, 0)), 1);
    assertEquals(Code.eval(Array.of(Symbol.NOT, 1)), 0);
    assertEquals(
        Code.eval(Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty()))), Array.of(1));
    assertEquals(
        Code.eval(Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty()))), List.of(1));
    assertEquals(
        Code.eval(
            Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty())))),
        1);
    assertEquals(
        Code.eval(
            Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty())))),
        List.empty());
  }
}
