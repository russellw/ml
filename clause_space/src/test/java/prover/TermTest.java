package prover;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

public class TermTest {
  @Test
  public void match() {

    // Subset of unify (see below)
    // Gives different results in several cases
    // In particular, has no notion of an occurs check
    // Assumes the inputs have disjoint variables
    var a = new Function("a");
    var b = new Function("b");
    var f = new Function("f");
    var g = new Function("g");
    var x = new Variable();
    var y = new Variable();
    var z = new Variable();
    Map<Variable, Term> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(a.match(a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(a.match(b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(x.match(x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertFalse(a.match(x, map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(x.match(y, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(f.call(a, x).match(f.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(f.call(a).match(g.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(f.call(x).match(f.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(f.call(x).match(g.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(f.call(x).unify(f.call(y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertFalse(f.call(g.call(x)).match(f.call(y), map));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertFalse(f.call(g.call(x), x).match(f.call(y, a), map));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertTrue(x.match(f.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(x.match(y, map));
    assertTrue(y.match(a, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertFalse(a.match(y, map));

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(x.match(a, map));
    assertFalse(b.match(x, map));
  }

  @Test
  public void unify() {

    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    var a = new Function("a");
    var b = new Function("b");
    var f = new Function("f");
    var g = new Function("g");
    var x = new Variable();
    var y = new Variable();
    var z = new Variable();
    Map<Variable, Term> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(a.unify(a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(a.unify(b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(x.unify(x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertTrue(a.unify(x, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), a);

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(x.unify(y, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(f.call(a, x).unify(f.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(f.call(a).unify(g.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(f.call(x).unify(f.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(f.call(x).unify(g.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(f.call(x).unify(f.call(y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertTrue(f.call(g.call(x)).unify(f.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(y.replace(map), g.call(x));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertTrue(f.call(g.call(x), x).unify(f.call(y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), g.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(x.unify(f.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(x.unify(y, map));
    assertTrue(y.unify(a, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertTrue(a.unify(y, map));
    assertTrue(x.unify(y, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(x.unify(a, map));
    assertFalse(b.unify(x, map));
  }
}
