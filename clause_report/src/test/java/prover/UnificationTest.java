package prover;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;
import io.vavr.collection.Map;
import org.junit.Test;

public class UnificationTest {
  @Test
  public void unify() {
    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f1 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f1");
    var f2 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f2");
    var g1 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "g1");
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
    map = Unification.unify(f2.call(a, x), f2.call(a, b), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), b);

    // f and g1 do not match
    map = Unification.unify(f1.call(a), g1.call(a), HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Unification.unify(f1.call(x), f1.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // f and g1 do not match
    map = Unification.unify(f1.call(x), g1.call(y), HashMap.empty());
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Unification.unify(f1.call(x), f2.call(y, z), HashMap.empty());
    assertNull(map);

    // Unifies y with the term g1(x)
    map = Unification.unify(f1.call(g1.call(x)), f1.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(y, map), g1.call(x));

    // Unifies x with constant a, and y with the term g1(a)
    map = Unification.unify(f2.call(g1.call(x), x), f2.call(y, a), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 2);
    assertEquals(Etc.replace(x, map), a);
    assertEquals(Etc.replace(y, map), g1.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Unification.unify(x, f1.call(x), HashMap.empty());
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
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f1 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f1");
    var f2 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "f2");
    var g1 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "g1");
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
    map = Unification.match(f2.call(a, x), f2.call(a, b), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), b);

    // f and g do not match
    map = Unification.match(f1.call(a), g1.call(a), HashMap.empty());
    assertNull(map);

    // x and y are aliased
    map = Unification.match(f1.call(x), f1.call(y), HashMap.empty());
    assertNotNull(map);
    assertEquals(map.size(), 1);
    assertEquals(Etc.replace(x, map), Etc.replace(y, map));

    // f and g do not match
    map = Unification.match(f1.call(x), g1.call(y), HashMap.empty());
    assertNull(map);

    // Fails. The f function symbols have different arity
    map = Unification.match(f1.call(x), f2.call(y, z), HashMap.empty());
    assertNull(map);

    // Unifies y with the term g(x)
    map = Unification.match(f1.call(g1.call(x)), f1.call(y), HashMap.empty());
    assertNull(map);

    // Unifies x with constant a, and y with the term g(a)
    map = Unification.match(f2.call(g1.call(x), x), f2.call(y, a), HashMap.empty());
    assertNull(map);

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = Unification.match(x, f1.call(x), HashMap.empty());
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
