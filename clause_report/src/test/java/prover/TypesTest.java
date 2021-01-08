package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import org.junit.Test;

public class TypesTest {
  @Test
  public void typeof() {
    assertEquals(Types.typeof(false), Symbol.BOOLEAN);
    assertEquals(Types.typeof(true), Symbol.BOOLEAN);
    assertEquals(Types.typeof(BigInteger.ZERO), Symbol.INTEGER);
    assertEquals(Types.typeof(BigRational.ZERO), Symbol.RATIONAL);
    assertEquals(Types.typeof(new Variable(Symbol.INDIVIDUAL)), Symbol.INDIVIDUAL);
  }
}
