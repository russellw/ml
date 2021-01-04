package prover;

import java.util.*;
import java.util.concurrent.TimeoutException;

public final class Subsumption {
  private static int steps;

  private static Map<Variable, Term> search(
      Term[] c, Term[] c2, Term[] d, Term[] d2, Map<Variable, Term> map) throws TimeoutException {
    if (steps == 1_000) throw new TimeoutException();
    steps++;

    // Matched everything in one polarity
    if (c.length == 0) {
      // Matched everything in the other polarity
      if (c2 == null) {
        return map;
      }

      // Try the other polarity
      return search(c2, null, d2, null, map);
    }

    // Try matching literals
    for (var ci = 0; ci < c.length; ci++) {
      Term[] c1 = null;
      var ce = Eq.of(c[ci]);
      for (var di = 0; di < d.length; di++) {
        Term[] d1 = null;
        var de = Eq.of(d[di]);

        // Search means preserve the original map
        // in case the search fails and need to backtrack
        Map<Variable, Term> m;

        // Try orienting equation one way
        m = new HashMap<>(map);
        if (Unification.match(ce.left, de.left, m) && Unification.match(ce.right, de.right, m)) {
          if (c1 == null) c1 = Term.remove(c, ci);
          d1 = Term.remove(d, di);
          m = search(c1, c2, d1, d2, m);
          if (m != null) return m;
        }

        // And the other way
        m = new HashMap<>(map);
        if (Unification.match(ce.left, de.right, m) && Unification.match(ce.right, de.left, m)) {
          if (c1 == null) c1 = Term.remove(c, ci);
          if (d1 == null) d1 = Term.remove(d, di);
          m = search(c1, c2, d1, d2, m);
          if (m != null) return m;
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
    if (c2.length < c1.length) {
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
      return search(c1, c2, d1, d2, new HashMap<>()) != null;
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
