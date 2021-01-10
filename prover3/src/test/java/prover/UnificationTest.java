package prover;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public class UnificationTest {
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
    assertTrue(Unification.match(a, a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(Unification.match(a, b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Unification.match(x, x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertFalse(Unification.match(a, x, map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.match(x, y, map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), Unification.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(Unification.match(f2.call(a, x), f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Unification.match(f1.call(a), g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.match(f1.call(x), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), Unification.replace(y, map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Unification.match(f1.call(x), g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Unification.match(f1.call(x), f2.call(y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertFalse(Unification.match(f1.call(g1.call(x)), f1.call(y), map));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertFalse(Unification.match(f2.call(g1.call(x), x), f2.call(y, a), map));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertTrue(Unification.match(x, f1.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(Unification.match(x, y, map));
    assertTrue(Unification.match(y, a, map));
    assertEquals(map.size(), 2);
    assertEquals(Unification.replace(x, map), a);
    assertEquals(Unification.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertFalse(Unification.match(a, y, map));

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(Unification.match(x, a, map));
    assertFalse(Unification.match(b, x, map));
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
    assertTrue(Unification.unify(a, a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(Unification.unify(a, b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(Unification.unify(x, x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertTrue(Unification.unify(a, x, map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), a);

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.unify(x, y, map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), Unification.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(Unification.unify(f2.call(a, x), f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), b);

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Unification.unify(f1.call(a), g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.unify(f1.call(x), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(x, map), Unification.replace(y, map));

    // f and g1 do not match
    map = new HashMap<>();
    assertFalse(Unification.unify(f1.call(x), g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Unification.unify(f1.call(x), f2.call(y, z), map));

    // Unifies y with the term g1(x)
    map = new HashMap<>();
    assertTrue(Unification.unify(f1.call(g1.call(x)), f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(Unification.replace(y, map), g1.call(x));

    // Unifies x with constant a, and y with the term g1(a)
    map = new HashMap<>();
    assertTrue(Unification.unify(f2.call(g1.call(x), x), f2.call(y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(Unification.replace(x, map), a);
    assertEquals(Unification.replace(y, map), g1.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(Unification.unify(x, f1.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(Unification.unify(x, y, map));
    assertTrue(Unification.unify(y, a, map));
    assertEquals(map.size(), 2);
    assertEquals(Unification.replace(x, map), a);
    assertEquals(Unification.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertTrue(Unification.unify(a, y, map));
    assertTrue(Unification.unify(x, y, map));
    assertEquals(map.size(), 2);
    assertEquals(Unification.replace(x, map), a);
    assertEquals(Unification.replace(y, map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(Unification.unify(x, a, map));
    assertFalse(Unification.unify(b, x, map));
  }
}
