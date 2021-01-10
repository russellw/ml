package prover;

import io.vavr.collection.HashMap;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.util.Collection;
import java.util.concurrent.TimeoutException;

public final class Subsumption {
  private static int steps;

  private Subsumption() {}

  private static Map<Variable, Object> search(
      Seq<Object> c, Seq<Object> c2, Seq<Object> d, Seq<Object> d2, Map<Variable, Object> map)
      throws TimeoutException {
    if (steps == 1_000) throw new TimeoutException();
    steps++;

    // Matched everything in one polarity
    if (c.isEmpty()) {
      // Matched everything in the other polarity
      if (c2 == null) {
        return map;
      }

      // Try the other polarity
      return search(c2, null, d2, null, map);
    }

    // Try matching literals
    for (var ci = 0; ci < c.size(); ci++) {
      Seq<Object> c1 = null;
      var ce = c.get(ci);
      for (var di = 0; di < d.size(); di++) {
        Seq<Object> d1 = null;
        var de = d.get(di);

        // Try orienting equation one way
        var m = Unification.match(Equality.left(ce), Equality.left(de), map);
        if (m != null) {
          m = Unification.match(Equality.right(ce), Equality.right(de), m);
          if (m != null) {
            if (c1 == null) c1 = c.removeAt(ci);
            d1 = d.removeAt(di);
            m = search(c1, c2, d1, d2, m);
            if (m != null) return m;
          }
        }

        // And the other way
        m = Unification.match(Equality.left(ce), Equality.right(de), map);
        if (m != null) {
          m = Unification.match(Equality.right(ce), Equality.left(de), m);
          if (m != null) {
            if (c1 == null) c1 = c.removeAt(ci);
            if (d1 == null) d1 = d.removeAt(di);
            m = search(c1, c2, d1, d2, m);
            if (m != null) return m;
          }
        }
      }
    }

    // No match
    return null;
  }

  public static boolean subsumes(Clause c, Clause d) {
    var variables = c.variables();
    variables.retainAll(d.variables());
    assert variables.isEmpty();

    var c1 = c.negative();
    var c2 = c.positive();
    var d1 = d.negative();
    var d2 = d.positive();

    // Fewer literals typically fail faster
    if (c2.size() < c1.size()) {
      // Swap negative and positive
      var ct = c1;
      c1 = c2;
      c2 = ct;

      // And in the other clause
      var dt = d1;
      d1 = d2;
      d2 = dt;
    }

    // Worst-case time is exponential
    // so give up if taking too long
    steps = 0;
    try {
      return search(c1, c2, d1, d2, HashMap.empty()) != null;
    } catch (TimeoutException e) {
      return false;
    }
  }

  public static boolean subsumesForward(Collection<Clause> clauses, Clause c) {
    for (var d : clauses) {
      if (d.subsumed) continue;
      if (subsumes(d, c)) return true;
    }
    return false;
  }

  public static void subsumeBackward(Clause c, Collection<Clause> clauses) {
    for (var d : clauses) {
      if (d.subsumed) continue;
      if (subsumes(c, d)) d.subsumed = true;
    }
  }
}
