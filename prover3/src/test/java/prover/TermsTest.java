package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.*;
import org.junit.Test;

public class TermsTest {
  @Test
  public void match() {
    // Subset of unify
    // Gives different results in several cases
    // In particular, has no notion of an occurs check
    // Assumes the inputs have disjoint variables
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f1 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f1");
    var f2 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f2");
    var g1 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "g1");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    Map<Variable, Object> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Terms.match(a, a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(Terms.match(a, b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Terms.match(x, x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertFalse(Terms.match(a, x, map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.match(x, y, map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(Terms.match(List.of(f2, a, x), List.of(f2, a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Terms.match(List.of(f1, a), List.of(g1, a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.match(List.of(f1, x), List.of(f1, y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Terms.match(List.of(f1, x), List.of(g1, y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Terms.match(List.of(f1, x), List.of(f2, y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertFalse(Terms.match(List.of(f1, List.of(g1, x)), List.of(f1, y), map));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertFalse(Terms.match(List.of(f2, List.of(g1, x), x), List.of(f2, y, a), map));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertTrue(Terms.match(x, List.of(f1, x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(Terms.match(x, y, map));
    assertTrue(Terms.match(y, a, map));
    assertEquals(map.size(), 2);
    assertEquals(Terms.replace(x, map), a);
    assertEquals(Terms.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertFalse(Terms.match(a, y, map));

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(Terms.match(x, a, map));
    assertFalse(Terms.match(b, x, map));
  }

  @Test
  public void unify() {
    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f1 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f1");
    var f2 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f2");
    var g1 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "g1");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    Map<Variable, Object> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Terms.unify(a, a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(Terms.unify(a, b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Terms.unify(x, x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertTrue(Terms.unify(a, x, map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), a);

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.unify(x, y, map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(Terms.unify(List.of(f2, a, x), List.of(f2, a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), b);

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Terms.unify(List.of(f1, a), List.of(g1, a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.unify(List.of(f1, x), List.of(f1, y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Terms.unify(List.of(f1, x), List.of(g1, y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Terms.unify(List.of(f1, x), List.of(f2, y, z), map));

    // Unifies y with the term g1(x)
    map = new HashMap<>();
    assertTrue(Terms.unify(List.of(f1, List.of(g1, x)), List.of(f1, y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(y, map), List.of(g1, x));

    // Unifies x with constant a, and y with the term g1(a)
    map = new HashMap<>();
    assertTrue(Terms.unify(List.of(f2, List.of(g1, x), x), List.of(f2, y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(Terms.replace(x, map), a);
    assertEquals(Terms.replace(y, map), List.of(g1, a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(Terms.unify(x, List.of(f1, x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(Terms.unify(x, y, map));
    assertTrue(Terms.unify(y, a, map));
    assertEquals(map.size(), 2);
    assertEquals(Terms.replace(x, map), a);
    assertEquals(Terms.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertTrue(Terms.unify(a, y, map));
    assertTrue(Terms.unify(x, y, map));
    assertEquals(map.size(), 2);
    assertEquals(Terms.replace(x, map), a);
    assertEquals(Terms.replace(y, map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(Terms.unify(x, a, map));
    assertFalse(Terms.unify(b, x, map));
  }

  @Test
  public void isomorphic() {
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "f");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    Map<Variable, Variable> map;

    // Atoms, equal
    map = new HashMap<>();
    assertTrue(Terms.isomorphic(a, a, map));
    assertEquals(map.size(), 0);

    // Atoms, unequal
    map = new HashMap<>();
    assertFalse(Terms.isomorphic(a, b, map));

    // Variables, equal
    map = new HashMap<>();
    assertTrue(Terms.isomorphic(x, x, map));
    assertEquals(map.size(), 0);

    // Variables, match
    map = new HashMap<>();
    assertTrue(Terms.isomorphic(x, y, map));
    assertEquals(map.size(), 2);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Terms.isomorphic(List.of(Symbol.EQUALS, a, a), List.of(Symbol.EQUALS, a, a), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Terms.isomorphic(List.of(Symbol.EQUALS, x, x), List.of(Symbol.EQUALS, x, x), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(
        Terms.isomorphic(
            List.of(Symbol.EQUALS, a, List.of(f, x)),
            List.of(Symbol.EQUALS, a, List.of(f, x)),
            map));
    assertEquals(map.size(), 0);

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Terms.isomorphic(List.of(Symbol.EQUALS, a, a), List.of(Symbol.EQUALS, a, b), map));

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Terms.isomorphic(List.of(Symbol.EQUALS, a, a), List.of(Symbol.EQUALS, a, x), map));

    // Compound, match
    map = new HashMap<>();
    assertTrue(
        Terms.isomorphic(
            List.of(Symbol.EQUALS, a, List.of(f, x)),
            List.of(Symbol.EQUALS, a, List.of(f, y)),
            map));
    assertEquals(map.size(), 2);
  }

  private static Set<Object> setOf(Object... q) {
    var r = new HashSet<>();
    Collections.addAll(r, q);
    return r;
  }

  @Test
  public void freeVariables() {
    var p1 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "p1");
    var p2 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p2");
    var p3 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p3");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    assertEquals(Terms.freeVariables(true), Collections.EMPTY_SET);
    assertEquals(Terms.freeVariables(List.of(p1, BigInteger.ONE)), Collections.EMPTY_SET);
    assertEquals(Terms.freeVariables(List.of(p1, x)), Collections.singleton(x));
    assertEquals(Terms.freeVariables(List.of(p2, x, x)), Collections.singleton(x));
    assertEquals(Terms.freeVariables(List.of(p2, x, y)), setOf(x, y));
    assertEquals(Terms.freeVariables(List.of(p3, x, y, z)), setOf(z, y, x));
    assertEquals(
        Terms.freeVariables(List.of(Symbol.ALL, List.of(), List.of(p3, x, y, z))), setOf(z, y, x));
    assertEquals(
        Terms.freeVariables(List.of(Symbol.ALL, List.of(x), List.of(p3, x, y, z))), setOf(z, y));
    assertEquals(
        Terms.freeVariables(List.of(Symbol.ALL, List.of(x, y), List.of(p3, x, y, z))), setOf(z));
  }

  @Test
  public void constant() {
    assertTrue(Terms.constant(false));
    assertTrue(Terms.constant(true));
    assertTrue(Terms.constant(List.of(Symbol.ADD, BigInteger.ONE, BigInteger.ONE)));
    var p = new Func(Symbol.BOOLEAN, "p");
    assertFalse(Terms.constant(p));
    var x = new Variable(Symbol.INTEGER);
    assertFalse(Terms.constant(List.of(Symbol.ADD, x, x)));
  }

  private void assertEval(Object a, Object b) {
    assertEquals(Terms.simplify(a), b);
  }

  @Test
  public void simplify() {
    var x = new Variable(Symbol.INTEGER);
    var y = new Variable(Symbol.INTEGER);
    assertEval(BigInteger.ZERO, BigInteger.ZERO);
    assertEval(x, x);

    // equals
    assertEval(List.of(Symbol.EQUALS, BigInteger.ZERO, BigInteger.ZERO), true);
    assertEval(List.of(Symbol.EQUALS, BigInteger.ZERO, BigInteger.ONE), false);
    assertEval(List.of(Symbol.EQUALS, x, x), true);
    assertEval(List.of(Symbol.EQUALS, x, y), List.of(Symbol.EQUALS, x, y));

    // add
    assertEval(
        List.of(Symbol.ADD, BigInteger.valueOf(1), BigInteger.valueOf(2)), BigInteger.valueOf(3));
    assertEval(
        List.of(Symbol.ADD, x, BigInteger.valueOf(2)),
        List.of(Symbol.ADD, x, BigInteger.valueOf(2)));
    assertEval(
        List.of(Symbol.ADD, BigInteger.valueOf(1), y),
        List.of(Symbol.ADD, BigInteger.valueOf(1), y));
    assertEval(List.of(Symbol.ADD, x, y), List.of(Symbol.ADD, x, y));
    assertEval(
        List.of(Symbol.ADD, BigRational.of("1/10"), BigRational.of("2/10")),
        BigRational.of("3/10"));
    assertEval(
        List.of(
            Symbol.ADD,
            List.of(Symbol.TO_REAL, BigRational.of("1/10")),
            List.of(Symbol.TO_REAL, BigRational.of("2/10"))),
        List.of(Symbol.TO_REAL, BigRational.of("3/10")));
    var z = new Variable(Symbol.REAL);
    assertEval(
        List.of(
            Symbol.ADD,
            List.of(Symbol.TO_REAL, z),
            List.of(Symbol.TO_REAL, BigRational.of("2/10"))),
        List.of(
            Symbol.ADD,
            List.of(Symbol.TO_REAL, z),
            List.of(Symbol.TO_REAL, BigRational.of("2/10"))));
    assertEval(
        List.of(
            Symbol.ADD,
            List.of(Symbol.TO_REAL, BigRational.of("1/10")),
            List.of(Symbol.TO_REAL, z)),
        List.of(
            Symbol.ADD,
            List.of(Symbol.TO_REAL, BigRational.of("1/10")),
            List.of(Symbol.TO_REAL, z)));
    assertEval(
        List.of(Symbol.ADD, z, List.of(Symbol.TO_REAL, BigRational.of("2/10"))),
        List.of(Symbol.ADD, z, List.of(Symbol.TO_REAL, BigRational.of("2/10"))));
    assertEval(
        List.of(Symbol.ADD, List.of(Symbol.TO_REAL, BigRational.of("1/10")), z),
        List.of(Symbol.ADD, List.of(Symbol.TO_REAL, BigRational.of("1/10")), z));

    // subtract
    assertEval(
        List.of(Symbol.SUBTRACT, BigInteger.valueOf(3), BigInteger.valueOf(1)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.SUBTRACT, BigRational.of("3/10"), BigRational.of("1/10")),
        BigRational.of("2/10"));

    // multiply
    assertEval(
        List.of(Symbol.MULTIPLY, BigInteger.valueOf(2), BigInteger.valueOf(5)),
        BigInteger.valueOf(10));
    assertEval(
        List.of(Symbol.MULTIPLY, BigRational.of("1/10"), BigRational.of("1/10")),
        BigRational.of("1/100"));

    // divide
    assertEval(
        List.of(Symbol.DIVIDE, BigRational.of("1"), BigRational.of("3")), BigRational.of("1/3"));

    // <
    assertEval(List.of(Symbol.LESS, BigInteger.valueOf(2), BigInteger.valueOf(1)), false);
    assertEval(List.of(Symbol.LESS, BigInteger.valueOf(1), BigInteger.valueOf(1)), false);
    assertEval(List.of(Symbol.LESS, BigInteger.valueOf(1), BigInteger.valueOf(2)), true);

    // <=
    assertEval(List.of(Symbol.LESS_EQ, BigInteger.valueOf(2), BigInteger.valueOf(1)), false);
    assertEval(List.of(Symbol.LESS_EQ, BigInteger.valueOf(1), BigInteger.valueOf(1)), true);
    assertEval(List.of(Symbol.LESS_EQ, BigInteger.valueOf(1), BigInteger.valueOf(2)), true);

    // divideTruncate
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(-1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(1));

    // remainderTruncate
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(-2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-2));

    // divideFloor
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorDiv(5, 3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorDiv(-5, 3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorDiv(5, -3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorDiv(-5, -3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(1));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(-2));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-2));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(1));

    // remainderFloor
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorMod(5, 3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorMod(-5, 3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorMod(5, -3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorMod(-5, -3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(1));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-1));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-2));

    // divideEuclidean
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigInteger.valueOf(7), BigInteger.valueOf(3)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigInteger.valueOf(7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(-2));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigInteger.valueOf(-7), BigInteger.valueOf(3)),
        BigInteger.valueOf(-3));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigInteger.valueOf(-7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(3));

    // remainderEuclidean
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigInteger.valueOf(7), BigInteger.valueOf(3)),
        BigInteger.valueOf(1));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigInteger.valueOf(7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(1));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigInteger.valueOf(-7), BigInteger.valueOf(3)),
        BigInteger.valueOf(2));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigInteger.valueOf(-7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(2));

    // divideTruncate
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigRational.of(5), BigRational.of(3)), BigRational.of(1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigRational.of(-5), BigRational.of(3)), BigRational.of(-1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigRational.of(5), BigRational.of(-3)), BigRational.of(-1));
    assertEval(
        List.of(Symbol.DIVIDE_TRUNCATE, BigRational.of(-5), BigRational.of(-3)), BigRational.of(1));

    // remainderTruncate
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigRational.of(5), BigRational.of(3)),
        BigRational.of(2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigRational.of(-5), BigRational.of(3)),
        BigRational.of(-2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigRational.of(5), BigRational.of(-3)),
        BigRational.of(2));
    assertEval(
        List.of(Symbol.REMAINDER_TRUNCATE, BigRational.of(-5), BigRational.of(-3)),
        BigRational.of(-2));

    // divideFloor
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(5), BigRational.of(3)),
        BigRational.of(Math.floorDiv(5, 3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(-5), BigRational.of(3)),
        BigRational.of(Math.floorDiv(-5, 3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(5), BigRational.of(-3)),
        BigRational.of(Math.floorDiv(5, -3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(-5), BigRational.of(-3)),
        BigRational.of(Math.floorDiv(-5, -3)));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(5), BigRational.of(3)), BigRational.of(1));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(-5), BigRational.of(3)), BigRational.of(-2));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(5), BigRational.of(-3)), BigRational.of(-2));
    assertEval(
        List.of(Symbol.DIVIDE_FLOOR, BigRational.of(-5), BigRational.of(-3)), BigRational.of(1));

    // remainderFloor
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(5), BigRational.of(3)),
        BigRational.of(Math.floorMod(5, 3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(-5), BigRational.of(3)),
        BigRational.of(Math.floorMod(-5, 3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(5), BigRational.of(-3)),
        BigRational.of(Math.floorMod(5, -3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(-5), BigRational.of(-3)),
        BigRational.of(Math.floorMod(-5, -3)));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(5), BigRational.of(3)), BigRational.of(2));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(-5), BigRational.of(3)), BigRational.of(1));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(5), BigRational.of(-3)), BigRational.of(-1));
    assertEval(
        List.of(Symbol.REMAINDER_FLOOR, BigRational.of(-5), BigRational.of(-3)),
        BigRational.of(-2));

    // divideEuclidean
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigRational.of(7), BigRational.of(3)), BigRational.of(2));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigRational.of(7), BigRational.of(-3)),
        BigRational.of(-2));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigRational.of(-7), BigRational.of(3)),
        BigRational.of(-3));
    assertEval(
        List.of(Symbol.DIVIDE_EUCLIDEAN, BigRational.of(-7), BigRational.of(-3)),
        BigRational.of(3));

    // remainderEuclidean
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigRational.of(7), BigRational.of(3)),
        BigRational.of(1));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigRational.of(7), BigRational.of(-3)),
        BigRational.of(1));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigRational.of(-7), BigRational.of(3)),
        BigRational.of(2));
    assertEval(
        List.of(Symbol.REMAINDER_EUCLIDEAN, BigRational.of(-7), BigRational.of(-3)),
        BigRational.of(2));

    // negate
    assertEval(List.of(Symbol.NEGATE, BigInteger.valueOf(3)), BigInteger.valueOf(-3));
    assertEval(List.of(Symbol.NEGATE, BigRational.of("3/10")), BigRational.of("-3/10"));

    // ceil
    assertEval(List.of(Symbol.CEIL, BigRational.of("0")), BigRational.of("0"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("1/10")), BigRational.of("1"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("5/10")), BigRational.of("1"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("9/10")), BigRational.of("1"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("-1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("-5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.CEIL, BigRational.of("-9/10")), BigRational.of("0"));

    // floor
    assertEval(List.of(Symbol.FLOOR, BigRational.of("0")), BigRational.of("0"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("9/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("-1/10")), BigRational.of("-1"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("-5/10")), BigRational.of("-1"));
    assertEval(List.of(Symbol.FLOOR, BigRational.of("-9/10")), BigRational.of("-1"));

    // round
    assertEval(List.of(Symbol.ROUND, BigRational.of("0")), BigRational.of("0"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("9/10")), BigRational.of("1"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("-1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("-5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.ROUND, BigRational.of("-9/10")), BigRational.of("-1"));

    // truncate
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("0")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("9/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("-1/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("-5/10")), BigRational.of("0"));
    assertEval(List.of(Symbol.TRUNCATE, BigRational.of("-9/10")), BigRational.of("0"));
  }
}
