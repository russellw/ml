package prover;

import static org.junit.Assert.*;

import java.util.ArrayList;
import org.junit.Test;

public class SubsumptionTest {
  @Test
  public void subsumes() {
    var a = new Function("a");
    var b = new Function("b");
    var p = new Function("p");
    var q = new Function("q");
    var x = new Variable(Type.INDIVIDUAL, "x");
    var y = new Variable(Type.INDIVIDUAL, "y");
    var negative = new ArrayList<Term>();
    var positive = new ArrayList<Term>();
    Clause c, d;

    // false <= false
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));

    // false <= p
    negative.clear();
    positive.clear();
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p <= p
    negative.clear();
    positive.clear();
    positive.add(p);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));

    // !p <= !p
    negative.clear();
    negative.add(p);
    positive.clear();
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p);
    positive.clear();
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));

    // p <= p | p
    negative.clear();
    positive.clear();
    positive.add(p);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p);
    positive.add(p);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p !<= !p
    negative.clear();
    positive.clear();
    positive.add(p);
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p);
    positive.clear();
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p | q <= q | p
    negative.clear();
    positive.clear();
    positive.add(p);
    positive.add(q);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(q);
    positive.add(p);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p | q <= p | q | p
    negative.clear();
    positive.clear();
    positive.add(p);
    positive.add(q);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p);
    positive.add(q);
    positive.add(p);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
    negative.clear();
    positive.clear();
    positive.add(p.call(a));
    positive.add(p.call(b));
    positive.add(q.call(a));
    positive.add(q.call(b));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a));
    positive.add(q.call(a));
    positive.add(p.call(b));
    positive.add(q.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x,y) <= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(p.call(x, y));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,x) !<= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(p.call(x, x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a, b));
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) <= p(y)
    negative.clear();
    positive.clear();
    positive.add(p.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
    negative.clear();
    positive.clear();
    positive.add(p.call(x));
    positive.add(p.call(a.call(x)));
    positive.add(p.call(a.call(a.call(x))));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(y));
    positive.add(p.call(a.call(y)));
    positive.add(p.call(a.call(a.call(y))));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a) <= p(a) | p(b)
    negative.clear();
    positive.clear();
    positive.add(p.call(x));
    positive.add(p.call(a));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a));
    positive.add(p.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) <= p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(p.call(x));
    positive.add(p.call(a.call(x)));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a.call(y)));
    positive.add(p.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(p.call(x));
    positive.add(p.call(a.call(x)));
    positive.add(p.call(a.call(a.call(x))));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a.call(a.call(y))));
    positive.add(p.call(a.call(y)));
    positive.add(p.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // (a = x) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(new Eq(a, x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(new Eq(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // (x = a) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(new Eq(x, a));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(new Eq(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(p.call(y));
    negative.add(p.call(x));
    positive.clear();
    positive.add(q.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p.call(a));
    negative.add(p.call(b));
    positive.clear();
    positive.add(q.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(p.call(x));
    negative.add(p.call(y));
    positive.clear();
    positive.add(q.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p.call(a));
    negative.add(p.call(b));
    positive.clear();
    positive.add(q.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,a(x)) !<= p(a(y),a(y))
    negative.clear();
    positive.clear();
    positive.add(p.call(x, a.call(x)));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p.call(a.call(y), a.call(y)));
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));
  }
}
