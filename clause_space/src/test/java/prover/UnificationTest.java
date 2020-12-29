package prover;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

public class UnificationTest {
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
    assertEquals(x.replace(map), a);

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.unify(x, y, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(Unification.unify(f.call(a, x), f.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), b);

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Unification.unify(f.call(a), g.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(Unification.unify(f.call(x), f.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replace(map), y.replace(map));

    // f and g do not match
    map = new HashMap<>();
    assertFalse(Unification.unify(f.call(x), g.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(Unification.unify(f.call(x), f.call(y, z), map));

    // Unifies y with the term g(x)
    map = new HashMap<>();
    assertTrue(Unification.unify(f.call(g.call(x)), f.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(y.replace(map), g.call(x));

    // Unifies x with constant a, and y with the term g(a)
    map = new HashMap<>();
    assertTrue(Unification.unify(f.call(g.call(x), x), f.call(y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), g.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(Unification.unify(x, f.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(Unification.unify(x, y, map));
    assertTrue(Unification.unify(y, a, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertTrue(Unification.unify(a, y, map));
    assertTrue(Unification.unify(x, y, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replace(map), a);
    assertEquals(y.replace(map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(Unification.unify(x, a, map));
    assertFalse(Unification.unify(b, x, map));
  }
}
