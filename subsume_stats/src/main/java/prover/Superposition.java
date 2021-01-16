package prover;

import java.util.*;

// The superposition calculus generates new clauses by three rules:
//
// Equality resolution
// c | c0 != c1
// ->
// c/map
// where
// map = unify(c0, c1)
//
// Equality factoring
// c | c0 = c1 | c2 = c3
// ->
// (c | c0 = c1 | c1 != c3)/map
// where
// map = unify(c0, c2)
//
// Superposition
// c | c0 = c1, d | d0(a) ?= d1
// ->
// (c | d | d0(c1) ?= d1)/map
// where
// map = unify(c0, a)
// a not variable
//
// This is a partial implementation of the superposition calculus
// A full implementation would also implement an order on equations
// e.g. lexicographic path ordering or Knuth-Bendix ordering
public final class Superposition {
  public final PriorityQueue<Clause> passive =
      new PriorityQueue<>(Comparator.comparingInt(Clause::volume));
  public final List<Clause> active = new ArrayList<>();

  private void clause(Clause c) {
    if (c.isTrue()) return;
    passive.add(c);
  }

  // Substitute and make new clause
  private void resolve(Clause c, int i, Map<Variable, Object> map) {
    // Negative literals
    var negative = new ArrayList<>(c.negativeSize - 1);
    for (var j = 0; j < c.negativeSize; j++)
      if (j != i) negative.add(Terms.replace(c.literals[j], map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize());
    for (var j = c.negativeSize; j < c.literals.length; j++)
      positive.add(Terms.replace(c.literals[j], map));

    // Make new clause
    clause(new Clause(negative, positive, Inference.RESOLVE, c));
  }

  // For each negative equation
  private void resolve(Clause c) {
    for (var i = 0; i < c.negativeSize; i++) {
      var e = c.literals[i];
      var map = new HashMap<Variable, Object>();
      if (Terms.unify(Equality.left(e), Equality.right(e), map)) resolve(c, i, map);
    }
  }

  // Substitute and make new clause
  private void factor(Clause c, Object c0, Object c1, int i, Object c2, Object c3) {
    if (!Equality.equatable(c1, c3)) return;
    var map = new HashMap<Variable, Object>();
    if (!Terms.unify(c0, c2, map)) return;

    // Negative literals
    var negative = new ArrayList<>(c.negativeSize + 1);
    for (var j = 0; j < c.negativeSize; j++) negative.add(Terms.replace(c.literals[j], map));
    negative.add(Terms.replace(Equality.of(c1, c3), map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize() - 1);
    for (var j = c.negativeSize; j < c.literals.length; j++)
      if (j != i) positive.add(Terms.replace(c.literals[j], map));

    // Make new clause
    clause(new Clause(negative, positive, Inference.FACTOR, c));
  }

  // For each positive equation (both directions) again
  private void factor(Clause c, int i, Object c0, Object c1) {
    for (var j = c.negativeSize; j < c.literals.length; j++) {
      if (j == i) continue;
      var e = c.literals[j];
      var c2 = Equality.left(e);
      var c3 = Equality.right(e);
      factor(c, c0, c1, j, c2, c3);
      factor(c, c0, c1, j, c3, c2);
    }
  }

  // For each positive equation (both directions)
  private void factor(Clause c) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = c.literals[i];
      var c0 = Equality.left(e);
      var c1 = Equality.right(e);
      factor(c, i, c0, c1);
      factor(c, i, c1, c0);
    }
  }

  // Check this subterm, substitute and make new clause
  private void superposition1(
      Clause c,
      Clause d,
      int ci,
      Object c0,
      Object c1,
      int di,
      Object d0,
      Object d1,
      List<Integer> position,
      Object a) {
    var map = new HashMap<Variable, Object>();
    if (!Terms.unify(c0, a, map)) return;
    var e = Equality.of(Etc.splice(d0, position, 0, c1), d1);

    // Negative literals
    var negative = new ArrayList<>(c.negativeSize + d.negativeSize);
    for (var i = 0; i < c.negativeSize; i++) negative.add(Terms.replace(c.literals[i], map));
    for (var i = 0; i < d.negativeSize; i++)
      if (i != di) negative.add(Terms.replace(d.literals[i], map));

    // Positive literals
    var positive = new ArrayList<>(c.positiveSize() + d.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.literals.length; i++)
      if (i != ci) positive.add(Terms.replace(c.literals[i], map));
    for (var i = d.negativeSize; i < d.literals.length; i++)
      if (i != di) positive.add(Terms.replace(d.literals[i], map));

    // Negative and positive superposition
    (di < d.negativeSize ? negative : positive).add(Terms.replace(e, map));

    // Make new clause
    clause(new Clause(negative, positive, Inference.SUPERPOSITION, c.original(), d.original()));
  }

  // Descend into subterms
  private void superposition(
      Clause c,
      Clause d,
      int ci,
      Object c0,
      Object c1,
      int di,
      Object d0,
      Object d1,
      List<Integer> position,
      Object a) {
    if (a instanceof Variable) return;
    superposition1(c, d, ci, c0, c1, di, d0, d1, position, a);
    if (!(a instanceof List)) return;
    var a1 = (List) a;
    for (var i = 1; i < a1.size(); i++) {
      position.add(i);
      superposition(c, d, ci, c0, c1, di, d0, d1, position, a1.get(i));
      position.remove(position.size() - 1);
    }
  }

  // For each equation in d (both directions)
  private void superposition(Clause c, Clause d, int ci, Object c0, Object c1) {
    if (c0 == Boolean.TRUE) return;
    for (var i = 0; i < d.literals.length; i++) {
      var e = d.literals[i];
      var d0 = Equality.left(e);
      var d1 = Equality.right(e);
      superposition(c, d, ci, c0, c1, i, d0, d1, new ArrayList<>(), d0);
      superposition(c, d, ci, c0, c1, i, d1, d0, new ArrayList<>(), d1);
    }
  }

  // For each positive equation in c (both directions)
  private void superposition(Clause c, Clause d) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = c.literals[i];
      var c0 = Equality.left(e);
      var c1 = Equality.right(e);
      superposition(c, d, i, c0, c1);
      superposition(c, d, i, c1, c0);
    }
  }
int givenLive;
int givenTotal;
  public void solve(Problem problem, long deadline) {
    passive.addAll(problem.clauses);
    problem.records.add(new Record(active, passive));
    while (!passive.isEmpty()) {
      // Given clause
      var g = passive.poll();
      if (g.subsumed) continue;

      // Solved
      if (g.isFalse()) {
        problem.refutation = g;
        problem.result = SZS.Unsatisfiable;
        return;
      }

      // Check resources
      if (System.currentTimeMillis() > deadline) {
        problem.result = SZS.Timeout;
        return;
      }

      // Rename variables for subsumption and subsequent inference
      var g1 = g.renameVariables();

      // Otter loop subsumes against passive clauses
      givenTotal++;
      if (Subsumption.subsumesForward(active, g1)) continue;
      //if (Subsumption.subsumesForward(passive, g1)) continue;
      givenLive++;
      Subsumption.subsumeBackward(g1, active);
      Subsumption.subsumeBackward(g1, passive);

      // Infer from one clause
      resolve(g);
      factor(g);

      // Sometimes need to match g with itself
      active.add(g);

      // Infer from two clauses
      for (var c : active) {
        if (c.subsumed) continue;
        superposition(c, g1);
        superposition(g1, c);
      }
      problem.records.add(new Record(active, passive));
    }

    // Superposition is not complete on arithmetic
    for (var c : active) {
      if (c.subsumed) continue;
      if (Etc.existsLeaf(
          c.term(),
          a -> {
            if (a instanceof List) return false;
            if (a instanceof Symbol) return false;
            return Types.isNumeric(Types.typeof(a));
          })) {
        problem.result = SZS.GaveUp;
        return;
      }
    }

    // Proof of satisfiability by running out of inferences
    // is rare, but can happen for very simple problems
    problem.result = SZS.Satisfiable;
  }
}
