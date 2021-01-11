package prover;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

public class SubsumptionTest {
  @Test
  public void subsumes() {
    var a = new Func(Symbol.INDIVIDUAL, "a");
    var a1 = new Func(List.of(Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "a1");
    var b = new Func(Symbol.INDIVIDUAL, "b");
    var p1 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "p1");
    var p2 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL, Symbol.INDIVIDUAL), "p2");
    var q1 = new Func(List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL), "q1");
    var x = new Variable(Symbol.INDIVIDUAL);
    var y = new Variable(Symbol.INDIVIDUAL);
    var negative = new ArrayList<>();
    var positive = new ArrayList<>();
    Clause c, d;

    // false <= false
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));

    // false <= p
    negative.clear();
    positive.clear();
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(p1);
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p <= p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(p1);
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));

    // !p <= !p
    negative.clear();
    negative.add(p1);
    positive.clear();
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    negative.add(p1);
    positive.clear();
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));

    // p <= p | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(p1);
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p !<= !p
    negative.clear();
    positive.clear();
    positive.add(p1);
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    negative.add(p1);
    positive.clear();
    d = new Clause(negative, positive, Inference.AXIOM);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p | q <= q | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(q1);
    positive.add(p1);
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p | q <= p | q | p
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(p1);
    positive.add(q1);
    positive.add(p1);
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, a));
    positive.add(List.of(p1, b));
    positive.add(List.of(q1, a));
    positive.add(List.of(q1, b));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, a));
    positive.add(List.of(q1, a));
    positive.add(List.of(p1, b));
    positive.add(List.of(q1, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x,y) <= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, x, y));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, a, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,x) !<= p(a,b)
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, x, x));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, a, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) <= p(y)
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, x));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, y));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, x));
    positive.add(List.of(p1, List.of(a1, x)));
    positive.add(List.of(p1, List.of(a1, List.of(a1, x))));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, y));
    positive.add(List.of(p1, List.of(a1, y)));
    positive.add(List.of(p1, List.of(a1, List.of(a1, y))));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a) <= p(a) | p(b)
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, x));
    positive.add(List.of(p1, a));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, a));
    positive.add(List.of(p1, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) <= p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, x));
    positive.add(List.of(p1, List.of(a1, x)));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, List.of(a1, y)));
    positive.add(List.of(p1, y));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, x));
    positive.add(List.of(p1, List.of(a1, x)));
    positive.add(List.of(p1, List.of(a1, List.of(a1, x))));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p1, List.of(a1, List.of(a1, y))));
    positive.add(List.of(p1, List.of(a1, y)));
    positive.add(List.of(p1, y));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertTrue(Subsumption.subsumes(d, c));

    // (a = x) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, x));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // (x = a) <= (a = b)
    negative.clear();
    positive.clear();
    positive.add(Equality.of(x, a));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(Equality.of(a, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(List.of(p1, y));
    negative.add(List.of(p1, x));
    positive.clear();
    positive.add(List.of(q1, x));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    negative.add(List.of(p1, a));
    negative.add(List.of(p1, b));
    positive.clear();
    positive.add(List.of(q1, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
    negative.clear();
    negative.add(List.of(p1, x));
    negative.add(List.of(p1, y));
    positive.clear();
    positive.add(List.of(q1, x));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    negative.add(List.of(p1, a));
    negative.add(List.of(p1, b));
    positive.clear();
    positive.add(List.of(q1, b));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertTrue(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));

    // p(x,a(x)) !<= p(a(y),a(y))
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, x, List.of(a1, x)));
    c = new Clause(negative, positive, Inference.AXIOM);
    negative.clear();
    positive.clear();
    positive.add(List.of(p2, List.of(a1, y), List.of(a1, y)));
    d = new Clause(negative, positive, Inference.AXIOM);
    assertFalse(Subsumption.subsumes(c, d));
    assertFalse(Subsumption.subsumes(d, c));
  }
}
