package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import org.junit.Test;

public class EtcTest {
  @Test
  public void divideEuclidean() {
    // int
    assertEquals(
        Etc.divideEuclidean(BigInteger.valueOf(7), BigInteger.valueOf(3)), BigInteger.valueOf(2));
    assertEquals(
        Etc.divideEuclidean(BigInteger.valueOf(7), BigInteger.valueOf(-3)), BigInteger.valueOf(-2));
    assertEquals(
        Etc.divideEuclidean(BigInteger.valueOf(-7), BigInteger.valueOf(3)), BigInteger.valueOf(-3));
    assertEquals(
        Etc.divideEuclidean(BigInteger.valueOf(-7), BigInteger.valueOf(-3)), BigInteger.valueOf(3));
  }

  @Test
  public void remainderEuclidean() {
    assertEquals(
        Etc.remainderEuclidean(BigInteger.valueOf(7), BigInteger.valueOf(3)),
        BigInteger.valueOf(1));
    assertEquals(
        Etc.remainderEuclidean(BigInteger.valueOf(7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(1));
    assertEquals(
        Etc.remainderEuclidean(BigInteger.valueOf(-7), BigInteger.valueOf(3)),
        BigInteger.valueOf(2));
    assertEquals(
        Etc.remainderEuclidean(BigInteger.valueOf(-7), BigInteger.valueOf(-3)),
        BigInteger.valueOf(2));
  }

  @Test
  public void divideFloor() {
    // Compare with expected values
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(5), BigInteger.valueOf(3)), BigInteger.valueOf(1));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(5), BigInteger.valueOf(-3)), BigInteger.valueOf(-2));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(-5), BigInteger.valueOf(3)), BigInteger.valueOf(-2));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(-5), BigInteger.valueOf(-3)), BigInteger.valueOf(1));

    // Compare with standard library int function
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorDiv(5, 3)));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorDiv(5, -3)));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorDiv(-5, 3)));
    assertEquals(
        Etc.divideFloor(BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorDiv(-5, -3)));
  }

  @Test
  public void remainderFloor() {
    // Compare with expected values
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(5), BigInteger.valueOf(3)), BigInteger.valueOf(2));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(5), BigInteger.valueOf(-3)), BigInteger.valueOf(-1));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(-5), BigInteger.valueOf(3)), BigInteger.valueOf(1));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(-5), BigInteger.valueOf(-3)), BigInteger.valueOf(-2));

    // Compare with standard library int function
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorMod(5, 3)));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorMod(5, -3)));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(-5), BigInteger.valueOf(3)),
        BigInteger.valueOf(Math.floorMod(-5, 3)));
    assertEquals(
        Etc.remainderFloor(BigInteger.valueOf(-5), BigInteger.valueOf(-3)),
        BigInteger.valueOf(Math.floorMod(-5, -3)));
  }
}
