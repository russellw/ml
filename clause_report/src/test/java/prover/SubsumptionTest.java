package prover;

import static org.junit.Assert.*;

import io.vavr.collection.Array;
import java.util.ArrayList;
import org.junit.Test;

public class SubsumptionTest {
  @Test
  public void subsumes() {
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var a1 = new Func(Array.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "a1");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var p1 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "p1");
    var p2 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p2");
    var q1 = new Func(Array.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "q1");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var negative = new ArrayList<>();
    var positive = new ArrayList<>();
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
    positive.add(p1);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p <= p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));

    // !p <= !p
    negative.clear();
    negative.add(p1);
    positive.clear();
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p1);
    positive.clear();
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));

    // p <= p | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(p1);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p !<= !p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p1);
    positive.clear();
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p | q <= q | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(q1);
    positive.add(p1);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p | q <= p | q | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    positive.add(p1);
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
    negative.clear();
    positive.clear();
    positive.add(p1.call(a));
    positive.add(p1.call(b));
    positive.add(q1.call(a));
    positive.add(q1.call(b));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(a));
    positive.add(q1.call(a));
    positive.add(p1.call(b));
    positive.add(q1.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x,y) <= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(p2.call(x, y));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p2.call(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,x) !<= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(p2.call(x, x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p2.call(a, b));
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) <= p(y)
    negative.clear();
    positive.clear();
    positive.add(p1.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
    negative.clear();
    positive.clear();
    positive.add(p1.call(x));
    positive.add(p1.call(a1.call(x)));
    positive.add(p1.call(a1.call(a1.call(x))));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(y));
    positive.add(p1.call(a1.call(y)));
    positive.add(p1.call(a1.call(a1.call(y))));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a) <= p(a) | p(b)
    negative.clear();
    positive.clear();
    positive.add(p1.call(x));
    positive.add(p1.call(a));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(a));
    positive.add(p1.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) <= p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(p1.call(x));
    positive.add(p1.call(a1.call(x)));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(a1.call(y)));
    positive.add(p1.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(p1.call(x));
    positive.add(p1.call(a1.call(x)));
    positive.add(p1.call(a1.call(a1.call(x))));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p1.call(a1.call(a1.call(y))));
    positive.add(p1.call(a1.call(y)));
    positive.add(p1.call(y));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // (a = x) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, x));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // (x = a) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(Equality.of(x, a));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(p1.call(y));
    negative.add(p1.call(x));
    positive.clear();
    positive.add(q1.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p1.call(a));
    negative.add(p1.call(b));
    positive.clear();
    positive.add(q1.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(p1.call(x));
    negative.add(p1.call(y));
    positive.clear();
    positive.add(q1.call(x));
    c = new Clause(negative, positive);
    negative.clear();
    negative.add(p1.call(a));
    negative.add(p1.call(b));
    positive.clear();
    positive.add(q1.call(b));
    d = new Clause(negative, positive);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,a(x)) !<= p(a(y),a(y))
    negative.clear();
    positive.clear();
    positive.add(p2.call(x, a1.call(x)));
    c = new Clause(negative, positive);
    negative.clear();
    positive.clear();
    positive.add(p2.call(a1.call(y), a1.call(y)));
    d = new Clause(negative, positive);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));
  }
}
