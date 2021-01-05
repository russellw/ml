package prover;

import static org.junit.Assert.*;

import io.vavr.collection.HashMap;
import io.vavr.collection.Map;
import org.junit.Test;

public class UnificationTest {
  @Test
  public void unify() {
    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    var a = new Func("a");
    var b = new Func("b");
    var f = new Func("f");
    var g = new Func("g");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    Map<Variable, Object> map;

    // Succeeds. (tautology)
    map = Unification.unify(a, a, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // a and b do not match
    map = Unification.unify(a, b, HashMap.empty());
    assertNull(map);

    // Succeeds. (tautology)
    map = Unification.unify(x, x, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = Unification.unify(a, x, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), a);

    // x and y are aliased
    map = Unification.unify(x, y, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = Unification.unify(f.call(a, x), f.call(a, b), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), b);

    // f and g do not match
    map = Unification.unify(f.call(a), g.call(a), HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Unification.unify(f.call(x), f.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // f and g do not match
    map = Unification.unify(f.call(x), g.call(y), HashMap.empty());
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Unification.unify(f.call(x), f.call(y, z), HashMap.empty());
    assertNull(map);

    // Unifies y with the term g(x)
    map = Unification.unify(f.call(g.call(x)), f.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(y, map), g.call(x));

    // Unifies x with constant a, and y with the term g(a)
    map = Unification.unify(f.call(g.call(x), x), f.call(y, a), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Etc.replace(x, map), a);
    assertEquals(Etc.replace(y, map), g.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Unification.unify(x, f.call(x), HashMap.empty());
    assertNull(map);

    // Both x and y are unified with the constant a
    map = Unification.unify(x, y, HashMap.empty());
    map = Unification.unify(y, a, map);
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Etc.replace(x, map), a);
    assertEquals(Etc.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = Unification.unify(a, y, HashMap.empty());
    map = Unification.unify(x, y, map);
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Etc.replace(x, map), a);
    assertEquals(Etc.replace(y, map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = Unification.unify(x, a, HashMap.empty());
    assertNotNull(map);
    map = Unification.unify(b, x, map);
    assertNull(map);
  }

  @Test
  public void match() {
    // Subset of unify (see below)
    // Gives different results in several cases
    // In particular, has no notion of an occurs check
    // Assumes the inputs have disjoint variables
    var a = new Func("a");
    var b = new Func("b");
    var f = new Func("f");
    var g = new Func("g");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    Map<Variable, Object> map;

    // Succeeds. (tautology)
    map = Unification.match(a, a, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // a and b do not match
    map = Unification.match(a, b, HashMap.empty());
    assertNull(map);

    // Succeeds. (tautology)
    map = Unification.match(x, x, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = Unification.match(a, x, HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Unification.match(x, y, HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // Function and constant symbols match, x is unified with the constant b
    map = Unification.match(f.call(a, x), f.call(a, b), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), b);

    // f and g do not match
    map = Unification.match(f.call(a), g.call(a), HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Unification.match(f.call(x), f.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // f and g do not match
    map = Unification.match(f.call(x), g.call(y), HashMap.empty());
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Unification.match(f.call(x), f.call(y, z), HashMap.empty());
    assertNull(map);

    // Unifies y with the term g(x)
    map = Unification.match(f.call(g.call(x)), f.call(y), HashMap.empty());
    assertNull(map);

    // Unifies x with constant a, and y with the term g(a)
    map = Unification.match(f.call(g.call(x), x), f.call(y, a), HashMap.empty());
    assertNull(map);

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Unification.match(x, f.call(x), HashMap.empty());
    assertNotNull(map);

    // Both x and y are unified with the constant a
    map = Unification.match(x, y, HashMap.empty());
    map = Unification.match(y, a, map);
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Etc.replace(x, map), a);
    assertEquals(Etc.replace(y, map), a);

    // As above (order of equations in set doesn't matter)
    map = Unification.match(a, y, HashMap.empty());
    assertNull(map);

    // Fails. a and b do not match, so x can't be unified with both
    map = Unification.match(x, a, HashMap.empty());
    assertNotNull(map);
    map = Unification.match(b, x, HashMap.empty());
    assertNull(map);
  }
}
