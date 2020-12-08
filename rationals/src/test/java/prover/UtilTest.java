package prover;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;

public class UtilTest {
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
    var rs = Util.cartesianProduct(qs);
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
  public void quote() {
    assertEquals(Util.quote('\'', "a"), "'a'");
    assertEquals(Util.quote('\'', "a'"), "'a\\''");
  }
}
