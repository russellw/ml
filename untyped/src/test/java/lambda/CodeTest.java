package lambda;

import static org.junit.Assert.*;

import io.vavr.collection.*;
import java.util.NoSuchElementException;
import org.junit.Test;

public class CodeTest {
  private static Object lambda(Object type, Object body) {
    return Array.of(Symbol.LAMBDA, new Variable(type), body);
  }

  @Test
  public void typeof() {
    assertEquals(Code.typeof(1), Symbol.INT);
    assertEquals(Code.typeof(true), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.ADD, 1, (Object) 2)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.SUB, 1, (Object) 2)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.MUL, 2, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.DIV, 10, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.REM, 10, (Object) 3)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.EQ, 10, 10)), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.LT, 1, (Object) 1)), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.LE, 1, (Object) 1)), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.AND, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.OR, false, false)), Symbol.BOOL);
    assertEquals(Code.typeof(Array.of(Symbol.NOT, false)), Symbol.BOOL);
    assertEquals(
        Code.typeof(Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty()))), Symbol.LIST);
    assertEquals(
        Code.typeof(
            Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty())))),
        Symbol.OBJECT);
    assertEquals(
        Code.typeof(
            Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty())))),
        Symbol.LIST);
    assertEquals(Code.typeof(Array.of(Symbol.CALL, lambda(Symbol.INT, 1), 2)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.CALL, lambda(Symbol.INT, 3), 2)), Symbol.INT);
    assertEquals(Code.typeof(Array.of(Symbol.IF, true, 1, 2)), Symbol.INT);
  }

  @Test
  public void rand() {
    var types = new Object[] {Symbol.BOOL, Symbol.INT};
    for (var type : types)
      for (var i = 0; i < 10000; i++)
        try {
          var a = Code.rand(List.empty(), type, 4);
          assertEquals(Code.typeof(a), type);
          var b = Code.simplify(HashMap.empty(), a);
          assertEquals(Code.typeof(b), type);
          // assert !(b instanceof Seq);
        } catch (ArithmeticException
            | GaveUp
            | NoSuchElementException
            | UnsupportedOperationException ignored) {
        }
  }

  @Test
  public void simplify() {
    var x = new Variable(Symbol.INT);
    var y = new Variable(Symbol.INT);
    assertEquals(Code.simplify(HashMap.empty(), 1), 1);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.ADD, 1, 2)), 3);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.SUB, 1, (Object) 2)), -1);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.MUL, 2, (Object) 3)), 6);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.DIV, 10, (Object) 3)), 3);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.REM, 10, (Object) 3)), 1);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.EQ, 10, 10)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.EQ, 10, 11)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LT, 1, (Object) 1)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LT, 1, (Object) 2)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LT, 2, (Object) 1)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LE, 1, (Object) 1)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LE, 1, (Object) 2)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.LE, 2, (Object) 1)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.AND, false, false)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.AND, false, true)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.AND, true, false)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.AND, true, true)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.OR, false, false)), false);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.OR, false, true)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.OR, true, false)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.OR, true, true)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.NOT, false)), true);
    assertEquals(Code.simplify(HashMap.empty(), Array.of(Symbol.NOT, true)), false);
    assertEquals(
        Code.simplify(HashMap.empty(), Array.of(Symbol.EQ, x, y)), Array.of(Symbol.EQ, x, y));
    assertEquals(
        Code.simplify(
            HashMap.empty(), Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty()))),
        Code.quote(Array.of(1)));
    assertEquals(
        Code.simplify(
            HashMap.empty(), Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty()))),
        Code.quote(List.of(1)));
    assertEquals(
        Code.simplify(
            HashMap.empty(),
            Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty())))),
        1);
    assertEquals(
        Code.simplify(
            HashMap.empty(),
            Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty())))),
        Code.quote(List.empty()));
  }

  @Test
  public void match() {
    // Subset of unify; tests adapted from unification tests
    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    // Gives different results in several cases
    // In particular, has no notion of an occurs check
    var a = "a";
    var b = "b";
    var f = "f";
    var g = "g";
    var x = new Variable(Symbol.INT);
    var y = new Variable(Symbol.INT);
    var z = new Variable(Symbol.INT);
    Map<Variable, Object> map;

    // Succeeds. (tautology)
    map = Code.match(HashMap.empty(), a, a);
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // a and b do not match
    map = Code.match(HashMap.empty(), a, b);
    assertNull(map);

    // Succeeds. (tautology)
    map = Code.match(HashMap.empty(), x, x);
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = Code.match(HashMap.empty(), a, x);
    assertNull(map);

    // x and y are aliased
    map = Code.match(HashMap.empty(), x, y);
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(map, x), Code.replace(map, y));

    // function and constant symbols match, x is unified with the constant b
    map =
        Code.match(HashMap.empty(), Array.of(Symbol.CALL, f, a, x), Array.of(Symbol.CALL, f, a, b));
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(map, x), b);

    // f and g do not match
    map = Code.match(HashMap.empty(), Array.of(Symbol.CALL, f, a), Array.of(Symbol.CALL, g, a));
    assertNull(map);

    // x and y are aliased
    map = Code.match(HashMap.empty(), Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, f, y));
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(map, x), Code.replace(map, y));

    // f and g do not match
    map = Code.match(HashMap.empty(), Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, g, y));
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Code.match(HashMap.empty(), Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, f, y, z));
    assertNull(map);

    // Unifies y with the term g(x)
    map =
        Code.match(
            HashMap.empty(),
            Array.of(Symbol.CALL, f, Array.of(Symbol.CALL, g, x)),
            Array.of(Symbol.CALL, f, y));
    assertNull(map);

    // Unifies x with constant a, and y with the term g(a)
    map =
        Code.match(
            HashMap.empty(),
            Array.of(Symbol.CALL, f, Array.of(Symbol.CALL, g, x), x),
            Array.of(Symbol.CALL, f, y, a));
    assertNull(map);

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Code.match(HashMap.empty(), x, Array.of(Symbol.CALL, f, x));
    assertNotNull(map);
    assertEquals(map.size(), 1);

    // Both x and y are unified with the constant a
    map = Code.match(HashMap.empty(), x, y);
    map = Code.match(map, y, a);
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Code.replace(map, x), a);
    assertEquals(Code.replace(map, y), a);

    // As above (order of equations in set doesn't matter)
    map = Code.match(HashMap.empty(), a, y);
    assertNull(map);

    // Fails. a and b do not match, so x can't be unified with both
    map = Code.match(HashMap.empty(), x, a);
    assertNotNull(map);
    map = Code.match(map, b, x);
    assertNull(map);
  }
}
