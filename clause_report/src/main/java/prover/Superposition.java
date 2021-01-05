package prover;

import io.vavr.collection.HashMap;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.PriorityQueue;

// The superposition calculus generates new clauses by three rules:
//
// Equality resolution
// c | c0 != c1
// ->
// c/s
// where
// s = unify(c0, c1)
//
// Equality factoring
// c | c0 = c1 | d0 = d1
// ->
// (c | c0 = c1 | c1 != d1)/s
// where
// s = unify(c0, d0)
//
// Superposition
// c | c0 = c1, d | d0(a) ?= d1
// ->
// (c | d | d0(c1) ?= d1)/s
// where
// s = unify(c0, a)
// a not variable
//
// This is a partial implementation of the superposition calculus
// A full implementation would also implement an order on equations
// e.g. lexicographic path ordering or Knuth-Bendix ordering
public final class Superposition {
  public static long timeout;
  public static PriorityQueue<Clause> unprocessed;
  public static ArrayList<Clause> processed;
  public static Clause proof;

  private Superposition() {}

  private static void clause(Clause c) {
    if (c.isTrue()) return;
    unprocessed.add(c);
  }

  // For each negative equation
  private static void resolution(Clause c) {
    for (var i = 0; i < c.negativeSize; i++) {
      var e = c.get(i);
      var map = Unification.unify(Equality.left(e), Equality.right(e), HashMap.empty());
      if (map != null) resolution(c, i, map);
    }
  }

  // Substitute and make new clause
  private static void resolution(Clause c, int ci, Map<Variable, Object> map) {
    // Negative literals
    var negative = new ArrayList<>(c.negativeSize - 1);
    for (var i = 0; i < c.negativeSize; i++) if (i != ci) negative.add(Etc.replace(c.get(i), map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize());
    for (var i = c.negativeSize; i < c.size(); i++) positive.add(Etc.replace(c.get(i), map));

    // Make new clause
    clause(new Clause(negative, positive));
  }

  // For each positive equation (both directions)
  private static void factoring(Clause c) {
    for (var i = c.negativeSize; i < c.size(); i++) {
      var e = c.get(i);
      factoring(c, i, Equality.left(e), Equality.right(e));
      factoring(c, i, Equality.right(e), Equality.left(e));
    }
  }

  // For each positive equation (both directions) again
  private static void factoring(Clause c, int ci, Object c0, Object c1) {
    for (var i = c.negativeSize; i < c.size(); i++) {
      if (i == ci) continue;
      var e = c.get(i);
      factoring(c, c0, c1, i, Equality.left(e), Equality.right(e));
      factoring(c, c0, c1, i, Equality.right(e), Equality.left(e));
    }
  }

  // Substitute and make new clause
  private static void factoring(Clause c, Object c0, Object c1, int di, Object d0, Object d1) {
    if (!Equality.equatable(c1, d1)) return;
    var map = Unification.unify(c0, d0, HashMap.empty());
    if (map == null) return;

    // Negative literals
    var negative = new ArrayList<>(c.negativeSize + 1);
    for (var i = 0; i < c.negativeSize; i++) negative.add(Etc.replace(c.get(i), map));
    negative.add(Etc.replace(Equality.of(c1, d1), map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.size(); i++)
      if (i != di) positive.add(Etc.replace(c.get(i), map));

    // Make new clause
    clause(new Clause(negative, positive));
  }

  // For each positive equation in c (both directions)
  private static void superposition(Clause c, Clause d) {
    for (var i = c.negativeSize; i < c.size(); i++) {
      var e = c.get(i);
      superposition(c, d, i, Equality.left(e), Equality.right(e));
      superposition(c, d, i, Equality.right(e), Equality.left(e));
    }
  }

  // For each equation in d (both directions)
  private static void superposition(Clause c, Clause d, int ci, Object c0, Object c1) {
    if (c0 == Boolean.TRUE) return;
    for (var i = 0; i < d.size(); i++) {
      var e = d.get(i);
      superposition(
          c,
          d,
          ci,
          c0,
          c1,
          i,
          Equality.left(e),
          Equality.right(e),
          new ArrayList<>(),
          Equality.left(e));
      superposition(
          c,
          d,
          ci,
          c0,
          c1,
          i,
          Equality.right(e),
          Equality.left(e),
          new ArrayList<>(),
          Equality.right(e));
    }
  }

  // Descend into subterms
  private static void superposition(
      Clause c,
      Clause d,
      int ci,
      Object c0,
      Object c1,
      int di,
      Object d0,
      Object d1,
      ArrayList<Integer> position,
      Object a) {
    if (a instanceof Variable) return;
    superposition1(c, d, ci, c0, c1, di, d0, d1, position, a);
    if (!(a instanceof Seq)) return;
    var a1 = (Seq) a;
    for (var i = 1; i < a1.size(); i++) {
      position.add(i);
      superposition(c, d, ci, c0, c1, di, d0, d1, position, a1.get(i));
      position.remove(position.size() - 1);
    }
  }

  // Check this subterm, substitute and make new clause
  private static void superposition1(
      Clause c,
      Clause d,
      int ci,
      Object c0,
      Object c1,
      int di,
      Object d0,
      Object d1,
      ArrayList<Integer> position,
      Object a) {
    var map = Unification.unify(c0, a, HashMap.empty());
    if (map == null) return;
    var e = Equality.of(Etc.splice(d0, position, 0, c1), d1);

    // Negative literals
    var negative = new ArrayList<>(c.negativeSize + d.negativeSize);
    for (var i = 0; i < c.negativeSize; i++) negative.add(Etc.replace(c.get(i), map));
    for (var i = 0; i < d.negativeSize; i++) if (i != di) negative.add(Etc.replace(d.get(i), map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize() + d.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.size(); i++)
      if (i != ci) positive.add(Etc.replace(c.get(i), map));
    for (var i = d.negativeSize; i < d.size(); i++)
      if (i != di) positive.add(Etc.replace(d.get(i), map));

    // Negative and positive superposition
    ((di < d.negativeSize) ? negative : positive).add(Etc.replace(e, map));

    // Make new clause
    clause(new Clause(negative, positive));
  }

  public static SZS satisfiable(Collection<Clause> clauses) {
    unprocessed = new PriorityQueue<>(Comparator.comparingInt(Clause::volume));
    unprocessed.addAll(clauses);
    processed = new ArrayList<>();
    proof = null;
    while (!unprocessed.isEmpty()) {
      // Given clause
      // Discount loop, given clause cannot have already been subsumed
      // Otter loop would check it for subsumption here
      var g = unprocessed.poll();

      // Solved
      if (g.isFalse()) {
        proof = g;
        return SZS.Unsatisfiable;
      }

      // Check resources
      if (System.currentTimeMillis() > timeout) return SZS.Timeout;

      // Rename variables for subsumption and subsequent inference
      var g1 = g.renameVariables();

      // Discount loop performed slightly better in tests
      // Otter loop would also subsume against unprocessed clauses
      if (Subsumption.subsumesForward(processed, g1)) continue;
      Subsumption.subsumeBackward(g1, processed);

      // Infer from one clause
      resolution(g);
      factoring(g);

      // Sometimes need to match g with itself
      processed.add(g);

      // Infer from two clauses
      for (var c : processed) {
        if (c.subsumed) continue;
        superposition(c, g1);
        superposition(g1, c);
      }
    }
    return SZS.Satisfiable;
  }
}
