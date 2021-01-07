package prover;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
    assertTrue(Etc.isomorphic(a, a, map));
    assertEquals(map.size(), 0);

    // Atoms, unequal
    map = new HashMap<>();
    assertFalse(Etc.isomorphic(a, b, map));

    // Variables, equal
    map = new HashMap<>();
    assertTrue(Etc.isomorphic(x, x, map));
    assertEquals(map.size(), 0);

    // Variables, match
    map = new HashMap<>();
    assertTrue(Etc.isomorphic(x, y, map));
    assertEquals(map.size(), 2);

    // Variables, different types
    map = new HashMap<>();
    assertFalse(Etc.isomorphic(x, r, map));

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Etc.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, a), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Etc.isomorphic(Array.of(Symbol.EQUALS, x, x), Array.of(Symbol.EQUALS, x, x), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(
        Etc.isomorphic(
            Array.of(Symbol.EQUALS, a, f.call(x)), Array.of(Symbol.EQUALS, a, f.call(x)), map));
    assertEquals(map.size(), 0);

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Etc.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, b), map));

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Etc.isomorphic(Array.of(Symbol.EQUALS, a, a), Array.of(Symbol.EQUALS, a, x), map));

    // Compound, match
    map = new HashMap<>();
    assertTrue(
        Etc.isomorphic(
            Array.of(Symbol.EQUALS, a, f.call(x)), Array.of(Symbol.EQUALS, a, f.call(y)), map));
    assertEquals(map.size(), 2);
  }

  @Test
  public void cartesianProduct() {
    ArrayList<List<String>> qs = new ArrayList<>();
    ArrayList<String> q;
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
    assertEquals(rs.get(i++), Arrays.asList("a0", "b0", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b0", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b0", "c2"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b0", "c3"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b1", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b1", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b1", "c2"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b1", "c3"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b2", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b2", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b2", "c2"));
    assertEquals(rs.get(i++), Arrays.asList("a0", "b2", "c3"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b0", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b0", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b0", "c2"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b0", "c3"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b1", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b1", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b1", "c2"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b1", "c3"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b2", "c0"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b2", "c1"));
    assertEquals(rs.get(i++), Arrays.asList("a1", "b2", "c2"));
    assertEquals(rs.get(i), Arrays.asList("a1", "b2", "c3"));
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
