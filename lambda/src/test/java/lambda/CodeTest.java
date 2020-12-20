package lambda;

import static org.junit.Assert.*;

import io.vavr.collection.*;
import java.util.NoSuchElementException;
import org.junit.Test;

public class CodeTest {
  private static Object lambda(Object type, Object body) {
    return Array.of(Symbol.LAMBDA, type, body);
  }

  @Test
  public void typeof() {
    var env = List.empty();
    assertEquals(Code.typeof(env, 1), Symbol.INT);
    assertEquals(Code.typeof(env, true), Symbol.BOOL);
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
    assertEquals(
        Code.typeof(env, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty()))),
        Symbol.LIST);
    assertEquals(
        Code.typeof(
            env,
            Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty())))),
        Symbol.OBJECT);
    assertEquals(
        Code.typeof(
            env,
            Array.of(Symbol.TAIL, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty())))),
        Symbol.LIST);
    assertEquals(Code.typeof(env, Array.of(Symbol.CALL, lambda(Symbol.INT, 1), 2)), Symbol.INT);
    assertEquals(
        Code.typeof(env, Array.of(Symbol.CALL, lambda(Symbol.INT, Array.of(Symbol.ARG, 0)), 2)),
        Symbol.INT);
    assertEquals(Code.typeof(env, Array.of(Symbol.IF, true, 1, 2)), Symbol.INT);
  }

  @Test
  public void rand() {
    var types = new Object[] {Symbol.BOOL, Symbol.INT};
    for (var type : types)
      for (var i = 0; i < 10000; i++)
        try {
          var a = Code.rand(List.empty(), type, 4);
          assertEquals(Code.typeof(List.empty(), a), type);
          var b = Code.simplify(List.empty(), a);
          assertEquals(Code.typeof(List.empty(), b), type);
          // assert !(b instanceof Seq);
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
    assertEquals(Code.simplify(List.empty(), 1), 1);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.ADD, 1, 2)), 3);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.SUB, 1, (Object) 2)), -1);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.MUL, 2, (Object) 3)), 6);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.DIV, 10, (Object) 3)), 3);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.REM, 10, (Object) 3)), 1);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.EQ, 10, 10)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.EQ, 10, 11)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LT, 1, (Object) 1)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LT, 1, (Object) 2)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LT, 2, (Object) 1)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LE, 1, (Object) 1)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LE, 1, (Object) 2)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.LE, 2, (Object) 1)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.AND, false, false)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.AND, false, true)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.AND, true, false)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.AND, true, true)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.OR, false, false)), false);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.OR, false, true)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.OR, true, false)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.OR, true, true)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.NOT, false)), true);
    assertEquals(Code.simplify(List.empty(), Array.of(Symbol.NOT, true)), false);
    assertEquals(
        Code.simplify(
            List.of(new Variable(Symbol.INT), new Variable(Symbol.INT)), Array.of(Symbol.EQ, x, y)),
        Array.of(Symbol.EQ, x, y));
    assertEquals(
        Code.simplify(List.empty(), Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty()))),
        Code.quote(Array.of(1)));
    assertEquals(
        Code.simplify(
            List.empty(), Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, Array.empty()))),
        Code.quote(List.of(1)));
    assertEquals(
        Code.simplify(
            List.empty(),
            Array.of(Symbol.HEAD, Array.of(Symbol.CONS, 1, Array.of(Symbol.QUOTE, List.empty())))),
        1);
    assertEquals(
        Code.simplify(
            List.empty(),
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
    map = Code.match(a, a, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // a and b do not match
    map = Code.match(a, b, HashMap.empty());
    assertNull(map);

    // Succeeds. (tautology)
    map = Code.match(x, x, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = Code.match(a, x, HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Code.match(x, y, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(x, map), Code.replace(y, map));

    // function and constant symbols match, x is unified with the constant b
    map =
        Code.match(Array.of(Symbol.CALL, f, a, x), Array.of(Symbol.CALL, f, a, b), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(x, map), b);

    // f and g do not match
    map = Code.match(Array.of(Symbol.CALL, f, a), Array.of(Symbol.CALL, g, a), HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Code.match(Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, f, y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Code.replace(x, map), Code.replace(y, map));

    // f and g do not match
    map = Code.match(Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, g, y), HashMap.empty());
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Code.match(Array.of(Symbol.CALL, f, x), Array.of(Symbol.CALL, f, y, z), HashMap.empty());
    assertNull(map);

    // Unifies y with the term g(x)
    map =
        Code.match(
            Array.of(Symbol.CALL, f, Array.of(Symbol.CALL, g, x)),
            Array.of(Symbol.CALL, f, y),
            HashMap.empty());
    assertNull(map);

    // Unifies x with constant a, and y with the term g(a)
    map =
        Code.match(
            Array.of(Symbol.CALL, f, Array.of(Symbol.CALL, g, x), x),
            Array.of(Symbol.CALL, f, y, a),
            HashMap.empty());
    assertNull(map);

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Code.match(x, Array.of(Symbol.CALL, f, x), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);

    // Both x and y are unified with the constant a
    map = Code.match(x, y, HashMap.empty());
    map = Code.match(y, a, map);
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Code.replace(x, map), a);
    assertEquals(Code.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = Code.match(a, y, HashMap.empty());
    assertNull(map);

    // Fails. a and b do not match, so x can't be unified with both
    map = Code.match(x, a, HashMap.empty());
    assertNotNull(map);
    map = Code.match(b, x, map);
    assertNull(map);
  }
}
