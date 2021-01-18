package prover;

import java.util.*;
import java.util.concurrent.TimeoutException;

public final class ParaSubsumption {
  private static final class DeterministicMatches {
    final Object[] c;
    final Object[] d;
    final Map<Variable, Object> map;

    DeterministicMatches(Object[] c, Object[] d, Map<Variable, Object> map) {
      this.c = c;
      this.d = d;
      this.map = map;
    }
  }

  private static int steps;

  public static boolean match(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Variable
    if (a instanceof Variable) {
      var a1 = (Variable) a;

      // Existing mapping
      var a2 = map.get(a1);
      if (a2 != null) return a2.equals(b);

      // New mapping
      map.put(a1, b);
      return true;
    }

    // Compounds
    if (a instanceof List) {
      var a1 = (List) a;
      assert a1.get(0) != Symbol.EQUALS;
      if (b instanceof List) {
        var b1 = (List) b;
        assert b1.get(0) != Symbol.EQUALS;
        int n = a1.size();
        if (n != b1.size()) return false;
        if (a1.get(0) != b1.get(0)) return false;
        for (var i = 1; i < n; i++) if (!match(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

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
      Object[] c, Object[] d, Map<Variable, Object> map) {
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
      if (!match(c[ci], d[di], map)) return null;
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

  private static Map<Variable, Object> search(
      Object[] c, Object[] c2, Object[] d, Object[] d2, Map<Variable, Object> map)
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
        Map<Variable, Object> m;

        // Try orienting equation one way
        m = new HashMap<>(map);
        if (match(Equality.left(ce), Equality.left(de), m)
            && match(Equality.right(ce), Equality.right(de), m)) {
          if (c1 == null) c1 = Etc.removeAt(c, ci);
          d1 = Etc.removeAt(d, di);
          m = search(c1, c2, d1, d2, m);
          if (m != null) return m;
        }

        // And the other way
        m = new HashMap<>(map);
        if (match(Equality.left(ce), Equality.right(de), m)
            && match(Equality.right(ce), Equality.left(de), m)) {
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
      Map<Variable, Object> map = new HashMap<>();

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
