package prover;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import java.math.BigInteger;
import java.util.Collections;
import java.util.HashSet;
import org.junit.Test;

public class VariableTest {
  private static HashSet<Object> setOf(Object... q) {
    var r = new HashSet<>();
    Collections.addAll(r, q);
    return r;
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
