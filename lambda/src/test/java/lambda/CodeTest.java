package lambda;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import org.junit.Test;

public class CodeTest {
  private static Object call(Object a, Object b, Object c) {
    return Array.of(Array.of(a, b), c);
  }

  private static Object lambda(Object type, Object body) {
    return Array.of(Symbol.LAMBDA, type, body);
  }

  @Test
  public void eval() {
    var env = List.empty();
    assertEquals(Code.eval(env, 0), 0);
    assertEquals(Code.eval(env, call(Symbol.ADD, 1, 2)), 3);
    assertEquals(Code.eval(env, call(Symbol.SUB, 1, 2)), -1);
    assertEquals(Code.eval(env, call(Symbol.MUL, 2, 3)), 6);
    assertEquals(Code.eval(env, call(Symbol.DIV, 10, 3)), 3);
    assertEquals(Code.eval(env, call(Symbol.REM, 10, 3)), 1);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, 10, 10)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, 10, 11)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, List.empty(), Array.empty())), true);
    assertEquals(Code.eval(env, call(Symbol.LT, 1, 1)), false);
    assertEquals(Code.eval(env, call(Symbol.LT, 1, 2)), true);
    assertEquals(Code.eval(env, call(Symbol.LT, 2, 1)), false);
    assertEquals(Code.eval(env, call(Symbol.LE, 1, 1)), true);
    assertEquals(Code.eval(env, call(Symbol.LE, 1, 2)), true);
    assertEquals(Code.eval(env, call(Symbol.LE, 2, 1)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.AND, false, false)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.AND, false, true)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.AND, true, false)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.AND, true, true)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.OR, false, false)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.OR, false, true)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.OR, true, false)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.OR, true, true)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.NOT, false)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.NOT, true)), false);
    assertEquals(Code.eval(env, call(Symbol.CONS, 1, List.empty())), Array.of(1));
    assertEquals(Code.eval(env, call(Symbol.CONS, 1, Array.empty())), List.of(1));
    assertEquals(Code.eval(env, Array.of(Symbol.HEAD, call(Symbol.CONS, 1, Array.empty()))), 1);
    assertEquals(
        Code.eval(env, Array.of(Symbol.TAIL, call(Symbol.CONS, 1, Array.empty()))), List.empty());
    assertEquals(Code.eval(env, Array.of(lambda(Symbol.INT, 1), 2)), 1);
    assertEquals(Code.eval(env, Array.of(lambda(Symbol.INT, Array.of(Symbol.ARG, 0)), 2)), 2);
    assertEquals(Code.eval(env, Array.of(Symbol.IF, true, 1, 2)), 1);
    assertEquals(Code.eval(env, Array.of(Symbol.IF, false, 1, 2)), 2);
  }

  @Test
  public void typeof() {
    var env = List.empty();
    assertEquals(Code.typeof(env, 1), Symbol.INT);
    assertEquals(Code.typeof(env, true), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.empty()), Symbol.LIST);
    assertEquals(Code.typeof(env, call(Symbol.ADD, 1, 2)), Symbol.INT);
    assertEquals(Code.typeof(env, call(Symbol.SUB, 1, 2)), Symbol.INT);
    assertEquals(Code.typeof(env, call(Symbol.MUL, 2, 3)), Symbol.INT);
    assertEquals(Code.typeof(env, call(Symbol.DIV, 10, 3)), Symbol.INT);
    assertEquals(Code.typeof(env, call(Symbol.REM, 10, 3)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.EQ, 10, 10)), Symbol.BOOL);
    assertEquals(Code.typeof(env, call(Symbol.LT, 1, 1)), Symbol.BOOL);
    assertEquals(Code.typeof(env, call(Symbol.LE, 1, 1)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.AND, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.OR, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.NOT, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, call(Symbol.CONS, 1, List.empty())), Symbol.LIST);
    assertEquals(
        Code.typeof(env, Array.of(Symbol.HEAD, call(Symbol.CONS, 1, Array.empty()))),
        Symbol.OBJECT);
    assertEquals(
        Code.typeof(env, Array.of(Symbol.TAIL, call(Symbol.CONS, 1, Array.empty()))), Symbol.LIST);
    assertEquals(Code.typeof(env, Array.of(lambda(Symbol.INT, 1), 2)), Symbol.INT);
    assertEquals(
        Code.typeof(env, Array.of(lambda(Symbol.INT, Array.of(Symbol.ARG, 0)), 2)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.IF, true, 1, 2)), Symbol.INT);
  }
}
