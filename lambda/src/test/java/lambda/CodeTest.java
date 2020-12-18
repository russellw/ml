package lambda;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import io.vavr.collection.List;
import java.util.NoSuchElementException;
import org.junit.Test;

public class CodeTest {
  private static Object lambda(Object type, Object body) {
    return Array.of(Symbol.LAMBDA, type, body);
  }

  @Test
  public void eval() {
    var env = List.empty();
    assertEquals(Code.eval(env, 0), 0);
    assertEquals(Code.eval(env, Array.of(Symbol.ADD, 1, (Object) 2)), 3);
    assertEquals(Code.eval(env, Array.of(Symbol.SUB, 1, (Object) 2)), -1);
    assertEquals(Code.eval(env, Array.of(Symbol.MUL, 2, (Object) 3)), 6);
    assertEquals(Code.eval(env, Array.of(Symbol.DIV, 10, (Object) 3)), 3);
    assertEquals(Code.eval(env, Array.of(Symbol.REM, 10, (Object) 3)), 1);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, 10, 10)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, 10, 11)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.EQ, List.empty(), Array.empty())), true);
    assertEquals(Code.eval(env, Array.of(Symbol.LT, 1, (Object) 1)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.LT, 1, (Object) 2)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.LT, 2, (Object) 1)), false);
    assertEquals(Code.eval(env, Array.of(Symbol.LE, 1, (Object) 1)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.LE, 1, (Object) 2)), true);
    assertEquals(Code.eval(env, Array.of(Symbol.LE, 2, (Object) 1)), false);
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
    assertEquals(Code.eval(env, Array.of(Symbol.CONS, 1, List.empty())), Array.of(1));
    assertEquals(Code.eval(env, Array.of(Symbol.CONS, 1, Array.empty())), List.of(1));
    assertEquals(Code.eval(env, Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.empty()))), 1);
    assertEquals(
        Code.eval(env, Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.empty()))),
        List.empty());
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
    assertEquals(Code.typeof(env, Array.of(Symbol.ADD, 1, (Object) 2)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.SUB, 1, (Object) 2)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.MUL, 2, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.DIV, 10, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.REM, 10, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.EQ, 10, 10)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.LT, 1, (Object) 1)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.LE, 1, (Object) 1)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.AND, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.OR, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.NOT, false)), Symbol.BOOL);
    assertEquals(Code.typeof(env, Array.of(Symbol.CONS, 1, List.empty())), Symbol.LIST);
    assertEquals(
        Code.typeof(env, Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.empty()))),
        Symbol.OBJECT);
    assertEquals(
        Code.typeof(env, Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.empty()))),
        Symbol.LIST);
    assertEquals(Code.typeof(env, Array.of(lambda(Symbol.INT, 1), 2)), Symbol.INT);
    assertEquals(
        Code.typeof(env, Array.of(lambda(Symbol.INT, Array.of(Symbol.ARG, 0)), 2)), Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.IF, true, 1, 2)), Symbol.INT);
  }

  @Test
  public void rand() {
    var types = new Object[] {Symbol.BOOL, Symbol.INT};
    var env = List.empty();
    for (var type : types)
      for (var i = 0; i < 1000; i++)
        try {
          var a = Code.rand(env, type, 4);
          assertEquals(Code.typeof(env, a), type);
          assertEquals(Code.typeof(env, Code.simplify(a)), type);
          assertEquals(Code.typeof(env, Code.eval(env, a)), type);
          assertEquals(Code.eval(env, a), Code.eval(env, Code.simplify(a)));
        } catch (ArithmeticException
            | GaveUp
            | NoSuchElementException
            | UnsupportedOperationException ignored) {
        }
  }

  @Test
  public void simplify() {
    var x = Array.of(Symbol.ARG, 1);
    var y = Array.of(Symbol.ARG, 0);
    assertEquals(Code.simplify(1), 1);
    // assertEquals(Code.simplify(Array.of(Symbol.EQ, x, x)), true);
    assertEquals(Code.simplify(Array.of(Symbol.EQ, x, y)), Array.of(Symbol.EQ, x, y));
  }
}
