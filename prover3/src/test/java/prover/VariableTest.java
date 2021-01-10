package prover;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import java.math.BigInteger;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import org.junit.Test;

public class VariableTest {
  private static HashSet<Object> setOf(Object... q) {
    var r = new HashSet<>();
    Collections.addAll(r, q);
    return r;
  }

  @Test
  public void isomorphic() {
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var f = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "f");
    var r = new Variable(Symbol.REAL);
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    HashMap<Variable, Variable> map;

    // Atoms, equal
    map = new HashMap<>();
    assertTrue(Variable.isomorphic(a, a, map));
    assertEquals(map.size(), 0);

    // Atoms, unequal
    map = new HashMap<>();
    assertFalse(Variable.isomorphic(a, b, map));

    // Variables, equal
    map = new HashMap<>();
    assertTrue(Variable.isomorphic(x, x, map));
    assertEquals(map.size(), 0);

    // Variables, match
    map = new HashMap<>();
    assertTrue(Variable.isomorphic(x, y, map));
    assertEquals(map.size(), 2);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(
        Variable.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, a), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(
        Variable.isomorphic(Array.of(Symbol.EQUALS, x, x), Array.of(Symbol.EQUALS, x, x), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(
        Variable.isomorphic(
            Array.of(Symbol.EQUALS, a, f.call(x)), Array.of(Symbol.EQUALS, a, f.call(x)), map));
    assertEquals(map.size(), 0);

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(
        Variable.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, b), map));

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(
        Variable.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, x), map));

    // Compound, match
    map = new HashMap<>();
    assertTrue(
        Variable.isomorphic(
            Array.of(Symbol.EQUALS, a, f.call(x)), Array.of(Symbol.EQUALS, a, f.call(y)), map));
    assertEquals(map.size(), 2);
  }

  @Test
  public void freeVariables() {
    var p1 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "p1");
    var p2 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p2");
    var p3 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p3");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var z = new Variable(Symbol.INDIVIDUAL);
    assertEquals(Variable.freeVariables(true), Collections.EMPTY_SET);
    assertEquals(Variable.freeVariables(Array.of(p1, BigInteger.ONE)), Collections.EMPTY_SET);
    assertEquals(Variable.freeVariables(Array.of(p1, x)), Collections.singleton(x));
    assertEquals(Variable.freeVariables(Array.of(p2, x, x)), Collections.singleton(x));
    assertEquals(Variable.freeVariables(Array.of(p2, x, y)), setOf(x, y));
    assertEquals(Variable.freeVariables(Array.of(p3, x, y, z)), setOf(z, y, x));
    assertEquals(
        Variable.freeVariables(Array.of(Symbol.ALL, Array.empty(), Array.of(p3, x, y, z))),
        setOf(z, y, x));
    assertEquals(
        Variable.freeVariables(Array.of(Symbol.ALL, Array.of(x), Array.of(p3, x, y, z))),
        setOf(z, y));
    assertEquals(
        Variable.freeVariables(Array.of(Symbol.ALL, Array.of(x, y), Array.of(p3, x, y, z))),
        setOf(z));
  }
}
