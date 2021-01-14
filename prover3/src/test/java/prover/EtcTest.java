package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

public class EtcTest {
  @Test
  public void divideEuclidean() {
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
  public void quote() {
    assertEquals(Etc.quote('\'', "a"), "'a'");
    assertEquals(Etc.quote('\'', "a'"), "'a\\''");
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
  public void cartesianProduct() {
    List<List<String>> qs = new ArrayList<>();
    List<String> q;
    q = new ArrayList<>();
    q.add("a0");
    q.add("a1");
    qs.add(q);
    q = new ArrayList<>();
    q.add("b0");
    q.add("b1");
    q.add("b2");
    qs.add(q);
    q = new ArrayList<>();
    q.add("c0");
    q.add("c1");
    q.add("c2");
    q.add("c3");
    qs.add(q);
    var rs = Etc.cartesianProduct(qs);
    var i = 0;
    assertEquals(rs.get(i++), List.of("a0", "b0", "c0"));
    assertEquals(rs.get(i++), List.of("a0", "b0", "c1"));
    assertEquals(rs.get(i++), List.of("a0", "b0", "c2"));
    assertEquals(rs.get(i++), List.of("a0", "b0", "c3"));
    assertEquals(rs.get(i++), List.of("a0", "b1", "c0"));
    assertEquals(rs.get(i++), List.of("a0", "b1", "c1"));
    assertEquals(rs.get(i++), List.of("a0", "b1", "c2"));
    assertEquals(rs.get(i++), List.of("a0", "b1", "c3"));
    assertEquals(rs.get(i++), List.of("a0", "b2", "c0"));
    assertEquals(rs.get(i++), List.of("a0", "b2", "c1"));
    assertEquals(rs.get(i++), List.of("a0", "b2", "c2"));
    assertEquals(rs.get(i++), List.of("a0", "b2", "c3"));
    assertEquals(rs.get(i++), List.of("a1", "b0", "c0"));
    assertEquals(rs.get(i++), List.of("a1", "b0", "c1"));
    assertEquals(rs.get(i++), List.of("a1", "b0", "c2"));
    assertEquals(rs.get(i++), List.of("a1", "b0", "c3"));
    assertEquals(rs.get(i++), List.of("a1", "b1", "c0"));
    assertEquals(rs.get(i++), List.of("a1", "b1", "c1"));
    assertEquals(rs.get(i++), List.of("a1", "b1", "c2"));
    assertEquals(rs.get(i++), List.of("a1", "b1", "c3"));
    assertEquals(rs.get(i++), List.of("a1", "b2", "c0"));
    assertEquals(rs.get(i++), List.of("a1", "b2", "c1"));
    assertEquals(rs.get(i++), List.of("a1", "b2", "c2"));
    assertEquals(rs.get(i), List.of("a1", "b2", "c3"));
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

  @Test
  public void withoutExtension() {
    assertEquals(Etc.withoutExtension("abc.txt"), "abc");
    assertEquals(Etc.withoutExtension("abc"), "abc");
  }

  @Test
  public void withoutDir() {
    assertEquals(Etc.withoutDir("abc.txt"), "abc.txt");
    assertEquals(Etc.withoutDir("abc"), "abc");
    assertEquals(Etc.withoutDir("/dir/abc"), "abc");
    assertEquals(Etc.withoutDir("\\dir\\abc"), "abc");
    assertEquals(Etc.withoutDir("C:\\dir\\abc"), "abc");
  }

  @Test
  public void exists() {
    assertTrue(Etc.exists(List.of('a', 'b', 'C'), c -> Character.isUpperCase((Character) c)));
    assertFalse(Etc.exists(List.of('a', 'b', 'c'), c -> Character.isUpperCase((Character) c)));
    assertFalse(Etc.exists(List.of(), c -> Character.isUpperCase((Character) c)));
  }

  @Test
  public void all() {
    assertTrue(Etc.all(List.of('A', 'B', 'C'), c -> Character.isUpperCase((Character) c)));
    assertFalse(Etc.all(List.of('a', 'b', 'C'), c -> Character.isUpperCase((Character) c)));
    assertFalse(Etc.all(List.of('a', 'b', 'c'), c -> Character.isUpperCase((Character) c)));
    assertTrue(Etc.all(List.of(), c -> Character.isUpperCase((Character) c)));
  }

  @Test
  public void count() {
    assertEquals(Etc.count(List.of('A', 'B', 'C'), c -> Character.isUpperCase((Character) c)), 3);
    assertEquals(Etc.count(List.of('a', 'b', 'C'), c -> Character.isUpperCase((Character) c)), 1);
    assertEquals(Etc.count(List.of('a', 'b', 'c'), c -> Character.isUpperCase((Character) c)), 0);
    assertEquals(Etc.count(List.of(), c -> Character.isUpperCase((Character) c)), 0);
  }
}
