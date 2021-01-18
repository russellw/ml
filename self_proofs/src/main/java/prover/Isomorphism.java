package prover;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;

public class Isomorphism {
  private static final class DeterministicMatches {
    final Object[] c;
    final Object[] d;
    final Map<Variable, Variable> map;

    DeterministicMatches(Object[] c, Object[] d, Map<Variable, Variable> map) {
      this.c = c;
      this.d = d;
      this.map = map;
    }
  }

  private static int steps;

  private static void uniqueCandidate(Map<Object, Integer> map, Object key, int value) {
    var old = map.get(key);
    if (old == null) {
      map.put(key, value);
      return;
    }
    if (old >= 0) map.put(key, -1);
  }

  private static Map<Object, Integer> uniques(Object[] c) {
    var map = new HashMap<Object, Integer>();
    for (var i = 0; i < c.length; i++) {
      var a = c[i];
      if (a instanceof List) {
        var a1 = (List) a;
        var op = a1.get(0);
        if (op != Symbol.EQUALS) uniqueCandidate(map, op, i);
        continue;
      }
      uniqueCandidate(map, a, i);
    }
    return map;
  }

  private static DeterministicMatches deterministicMatches(
      Object[] c, Object[] d, Map<Variable, Variable> map) {
    // Top-level occurrences of each symbol
    var cs = uniques(c);
    var ds = uniques(d);

    // Remember which literals deterministically matched
    var cmatched = new boolean[c.length];
    var dmatched = new boolean[d.length];

    // Symbols that only occur once in c
    for (var key : cs.keySet()) {
      var ci = cs.get(key);
      if (ci < 0) continue;
      assert !cmatched[ci];

      // And only once in d
      var di = ds.get(key);
      if (di == null || di < 0) continue;
      assert !dmatched[di];

      // Are matched deterministically if at all
      if (!Terms.isomorphic(c[ci], d[di], map)) return null;
      cmatched[ci] = true;
      dmatched[di] = true;
    }

    // How many matched?
    var matched = 0;
    for (var m : cmatched) if (m) matched++;

    // Unmatched c literals
    var c1 = new Object[c.length - matched];
    var j = 0;
    for (var i = 0; i < c.length; i++) if (!cmatched[i]) c1[j++] = c[i];
    assert j == c1.length;

    // Unmatched d literals
    var d1 = new Object[d.length - matched];
    j = 0;
    for (var i = 0; i < d.length; i++) if (!dmatched[i]) d1[j++] = d[i];
    assert j == d1.length;

    // Return multiple values as object
    return new DeterministicMatches(c1, d1, map);
  }

  private static Map<Variable, Variable> search(
      Object[] c, Object[] c2, Object[] d, Object[] d2, Map<Variable, Variable> map)
      throws TimeoutException {
    if (steps == 1_000) throw new TimeoutException();
    steps++;

    // Matched everything in one polarity
    if (c.length == 0) {
      // Matched everything in the other polarity
      if (c2 == null) return map;

      // Try the other polarity
      return search(c2, null, d2, null, map);
    }

    // Try matching literals
    for (var ci = 0; ci < c.length; ci++) {
      Object[] c1 = null;
      var ce = c[ci];
      for (var di = 0; di < d.length; di++) {
        Object[] d1 = null;
        var de = d[di];

        // Search means preserve the original map
        // in case the search fails
        // and need to backtrack
        Map<Variable, Variable> m;

        // Try orienting equation one way
        m = new HashMap<>(map);
        if (Terms.isomorphic(Equality.left(ce), Equality.left(de), m)
            && Terms.isomorphic(Equality.right(ce), Equality.right(de), m)) {
          if (c1 == null) c1 = Etc.removeAt(c, ci);
          d1 = Etc.removeAt(d, di);
          m = search(c1, c2, d1, d2, m);
          if (m != null) return m;
        }

        // And the other way
        m = new HashMap<>(map);
        if (Terms.isomorphic(Equality.left(ce), Equality.right(de), m)
            && Terms.isomorphic(Equality.right(ce), Equality.left(de), m)) {
          if (c1 == null) c1 = Etc.removeAt(c, ci);
          if (d1 == null) d1 = Etc.removeAt(d, di);
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
      Map<Variable, Variable> map = new HashMap<>();

      if (false) {
        // Negative literals (unless swapped)
        var dm = deterministicMatches(c1, d1, map);
        if (dm == null) {
          return false;
        }
        map = dm.map;
        c1 = dm.c;
        d1 = dm.d;

        // Positive literals (unless swapped)
        dm = deterministicMatches(c2, d2, map);
        if (dm == null) {
          return false;
        }
        map = dm.map;
        c2 = dm.c;
        d2 = dm.d;
      }

      // Search for nondeterministic matches
      map = search(c1, c2, d1, d2, map);
      return map != null;
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
