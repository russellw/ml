package prover;

import java.util.*;

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
  public static List<Clause> processed;
  public static Clause proof;

  private static void clause(Clause c) {
    if (c.isTrue()) return;
    unprocessed.add(c);
  }

  // For each negative equation
  private static void resolution(Clause c) {
    for (var i = 0; i < c.negativeSize; i++) {
      var e = Eq.of(c.literals[i]);
      var map = new HashMap<Var, Term>();
      if (Unification.unify(e.left, e.right, map)) resolution(c, i, map);
    }
  }

  // Substitute and make new clause
  private static void resolution(Clause c, int ci, Map<Var, Term> map) {
    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize - 1);
    for (var i = 0; i < c.negativeSize; i++) if (i != ci) negative.add(c.literals[i].replace(map));

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize());
    for (var i = c.negativeSize; i < c.literals.length; i++)
      positive.add(c.literals[i].replace(map));

    // Make new clause
    clause(new Clause(negative, positive));
  }

  // For each positive equation (both directions)
  private static void factoring(Clause c) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = Eq.of(c.literals[i]);
      factoring(c, i, e.left, e.right);
      factoring(c, i, e.right, e.left);
    }
  }

  // For each positive equation (both directions) again
  private static void factoring(Clause c, int ci, Term c0, Term c1) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      if (i == ci) continue;
      var e = Eq.of(c.literals[i]);
      factoring(c, c0, c1, i, e.left, e.right);
      factoring(c, c0, c1, i, e.right, e.left);
    }
  }

  // Substitute and make new clause
  private static void factoring(Clause c, Term c0, Term c1, int di, Term d0, Term d1) {
    if (!Eq.equatable(c1, d1)) return;
    var map = new HashMap<Var, Term>();
    if (!Unification.unify(c0, d0, map)) return;

    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize + 1);
    for (var i = 0; i < c.negativeSize; i++) negative.add(c.literals[i].replace(map));
    negative.add(new Eq(c1, d1).replace(map).term());

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.literals.length; i++)
      if (i != di) positive.add(c.literals[i].replace(map));

    // Make new clause
    clause(new Clause(negative, positive));
  }

  // For each positive equation in c (both directions)
  private static void superposition(Clause c, Clause d) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = Eq.of(c.literals[i]);
      superposition(c, d, i, e.left, e.right);
      superposition(c, d, i, e.right, e.left);
    }
  }

  // For each equation in d (both directions)
  private static void superposition(Clause c, Clause d, int ci, Term c0, Term c1) {
    if (c0 == Term.TRUE) return;
    for (var i = 0; i < d.literals.length; i++) {
      var e = Eq.of(d.literals[i]);
      superposition(c, d, ci, c0, c1, i, e.left, e.right, new ArrayList<>(), e.left);
      superposition(c, d, ci, c0, c1, i, e.right, e.left, new ArrayList<>(), e.right);
    }
  }

  // Descend into subterms
  private static void superposition(
      Clause c,
      Clause d,
      int ci,
      Term c0,
      Term c1,
      int di,
      Term d0,
      Term d1,
      List<Integer> position,
      Term a) {
    if (a instanceof Var) return;
    superposition1(c, d, ci, c0, c1, di, d0, d1, position, a);
    for (var i = 1; i < a.size(); i++) {
      position.add(i);
      superposition(c, d, ci, c0, c1, di, d0, d1, position, a.get(i));
      position.remove(position.size() - 1);
    }
  }

  // Check this subterm, substitute and make new clause
  private static void superposition1(
      Clause c,
      Clause d,
      int ci,
      Term c0,
      Term c1,
      int di,
      Term d0,
      Term d1,
      List<Integer> position,
      Term a) {
    var map = new HashMap<Var, Term>();
    if (!Unification.unify(c0, a, map)) return;
    var e = new Eq(d0.splice(position, c1), d1);

    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize + d.negativeSize);
    for (var i = 0; i < c.negativeSize; i++) negative.add(c.literals[i].replace(map));
    for (var i = 0; i < d.negativeSize; i++) if (i != di) negative.add(d.literals[i].replace(map));

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize() + d.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.literals.length; i++)
      if (i != ci) positive.add(c.literals[i].replace(map));
    for (var i = d.negativeSize; i < d.literals.length; i++)
      if (i != di) positive.add(d.literals[i].replace(map));

    // Negative and positive superposition
    ((di < d.negativeSize) ? negative : positive).add(e.replace(map).term());

    // Make new clause
    clause(new Clause(negative, positive));
  }

  public static Boolean satisfiable(Collection<Clause> clauses) {
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
        return false;
      }

      // Check resources
      if (System.currentTimeMillis() > timeout) return null;

      // Rename variables for subsumption and subsequent inference
      var g1 = g.renameVars();

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
    return true;
  }
}
