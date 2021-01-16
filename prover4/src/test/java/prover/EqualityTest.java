package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import org.junit.Test;

public class EqualityTest {
  @Test
  public void equatable() {
    assertTrue(Equality.equatable(BigInteger.ZERO, BigInteger.ONE));
    assertFalse(Equality.equatable(BigInteger.ZERO, true));
    assertFalse(Equality.equatable(BigInteger.ZERO, BigRational.ONE));
    var p = new Func(Symbol.BOOLEAN, "p");
    assertTrue(Equality.equatable(p, true));
    assertFalse(Equality.equatable(p, p));
  }
}
