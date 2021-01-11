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
    assertTrue(Terms.match(f2.call(a, x), f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Terms.match(f1.call(a), g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.match(f1.call(x), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Terms.match(f1.call(x), g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Terms.match(f1.call(x), f2.call(y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertFalse(Terms.match(f1.call(g1.call(x)), f1.call(y), map));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertFalse(Terms.match(f2.call(g1.call(x), x), f2.call(y, a), map));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertTrue(Terms.match(x, f1.call(x), map));

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
    assertTrue(Terms.unify(f2.call(a, x), f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), b);

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Terms.unify(f1.call(a), g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Terms.unify(f1.call(x), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(x, map), Terms.replace(y, map));

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Terms.unify(f1.call(x), g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Terms.unify(f1.call(x), f2.call(y, z), map));

    // Unifies y with the term g1(x)
    map = new HashMap<>();
    assertTrue(Terms.unify(f1.call(g1.call(x)), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Terms.replace(y, map), g1.call(x));

    // Unifies x with constant a, and y with the term g1(a)
    map = new HashMap<>();
    assertTrue(Terms.unify(f2.call(g1.call(x), x), f2.call(y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(Terms.replace(x, map), a);
    assertEquals(Terms.replace(y, map), g1.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(Terms.unify(x, f1.call(x), map));

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
            List.of(Symbol.EQUALS, a, f.call(x)), List.of(Symbol.EQUALS, a, f.call(x)), map));
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
            List.of(Symbol.EQUALS, a, f.call(x)), List.of(Symbol.EQUALS, a, f.call(y)), map));
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
}
