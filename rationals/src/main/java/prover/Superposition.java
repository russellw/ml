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
// Open questions:
// https://cs.stackexchange.com/questions/102957/superposition-calculus-elimination-of-redundant-atoms
// https://cs.stackexchange.com/questions/103556/superposition-calculus-greater-vs-greater-or-equal
public final class Superposition extends Thread {
  private Collection<Clause> input;

  // Main loop
  public final List<Clause> processed = new ArrayList<>();
  public final PriorityQueue<Clause> unprocessed =
      new PriorityQueue<>(Comparator.comparingInt(Superposition::volume));

  // Search space reduction
  private final LexicographicPathOrder lpo;
  public final Subsumption subsumption;

  // Result
  private boolean complete = true;
  public Clause conclusion;
  public SZS result;

  // Statistics
  public long generatedResolution;
  public long generatedFactoring;
  public long generatedSuperposition;
  public long tautologies;
  public long subsumedForward;
  public long subsumedBackward;
  public long timePrepEnd;

  public Superposition(Collection<Clause> clauses) {
    input = clauses;
    var ops = Clause.ops(clauses);
    lpo = new LexicographicPathOrder(ops);
    subsumption = new Subsumption(clauses);
  }

  private void clause(Clause c) {
    if (c.isTrue()) {
      tautologies++;
      return;
    }
    if (!subsumption.add(c)) {
      subsumedForward++;
      return;
    }
    c.setId();
    unprocessed.add(c);
  }

  private void clauseQuick(Clause c) {
    if (c.isTrue()) {
      tautologies++;
      return;
    }
    c.setId();
    subsumption.addQuick(c);
    unprocessed.add(c);
  }

  // For each positive equation (both directions)
  private void factoring(Clause c) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = Equation.of(c.literals[i]);
      factoring(c, i, e.left, e.right);
      factoring(c, i, e.right, e.left);
    }
  }

  // For each positive equation (both directions) again
  private void factoring(Clause c, int ci, Term c0, Term c1) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      if (i == ci) {
        continue;
      }
      var e = Equation.of(c.literals[i]);
      factoring(c, c0, c1, i, e.left, e.right);
      factoring(c, c0, c1, i, e.right, e.left);
    }
  }

  // Substitute and make new clause
  private void factoring(Clause c, Term c0, Term c1, int di, Term d0, Term d1) {
    if (!Equation.equatable(c1, d1)) {
      return;
    }
    var map = new HashMap<Variable, Term>();
    if (!c0.unify(d0, map)) {
      return;
    }

    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize + 1);
    for (var i = 0; i < c.negativeSize; i++) {
      negative.add(c.literals[i].replaceVars(map));
    }
    negative.add(Equation.of(c1, d1).replaceVars(map).term());

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      if (i != di) {
        positive.add(c.literals[i].replaceVars(map));
      }
    }

    // Make new clause
    generatedFactoring++;
    clause(new ClauseFactoring(negative, positive, c));
  }

  private static Clause freshVars(Clause clause) {
    var map = new HashMap<Variable, Variable>();
    var literals = Term.map(clause.literals, term -> term.freshVars(map));
    if (map.isEmpty()) {
      return clause;
    }
    return new ClauseRenamed(literals, clause.negativeSize, clause);
  }

  private static Clause original(Clause clause) {
    return (clause instanceof ClauseRenamed) ? ((Clause) clause.from()[0]) : clause;
  }

  // For each negative equation
  private void resolution(Clause c) {
    for (var i = 0; i < c.negativeSize; i++) {
      var e = Equation.of(c.literals[i]);
      var map = new HashMap<Variable, Term>();
      if (e.left.unify(e.right, map)) {
        resolution(c, i, map);
      }
    }
  }

  // Substitute and make new clause
  private void resolution(Clause c, int ci, Map<Variable, Term> map) {

    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize - 1);
    for (var i = 0; i < c.negativeSize; i++) {
      if (i != ci) {
        negative.add(c.literals[i].replaceVars(map));
      }
    }

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize());
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      positive.add(c.literals[i].replaceVars(map));
    }

    // Make new clause
    generatedResolution++;
    clause(new ClauseResolution(negative, positive, c));
  }

  @Override
  public void run() {

    // Superposition is not complete on arithmetic
    for (var c : input) {
      for (var a : c.literals) {
        a.walk(
            (b, depth) -> {
              switch (b.type().kind()) {
                case INTEGER:
                case RATIONAL:
                case REAL:
                  complete = false;
                  break;
              }
            });
      }
    }

    // Prepare
    for (var c : input) {
      if (Thread.interrupted()) {
        result = SZS.Timeout;
        timePrepEnd = System.currentTimeMillis();
        return;
      }
      clauseQuick(c);
    }
    timePrepEnd = System.currentTimeMillis();

    // Otter loop
    while (!unprocessed.isEmpty()) {

      // Given clause
      var g = unprocessed.poll();
      if (subsumption.subsumed(g)) {
        subsumedBackward++;
        continue;
      }
      processed.add(g);
      if (g.isFalse()) {
        result = SZS.Unsatisfiable;
        conclusion = g;
        return;
      }

      // Check resources
      if ((Main.clauseLimit > 0) && (processed.size() + unprocessed.size() > Main.clauseLimit)) {
        result = SZS.ResourceOut;
        return;
      }
      if (Thread.interrupted()) {
        result = SZS.Timeout;
        return;
      }

      // Infer from one clause
      resolution(g);
      factoring(g);

      // Sometimes need to match g with itself
      g = freshVars(g);

      // Infer from two clauses
      for (var c : processed) {
        if (subsumption.subsumed(c)) {
          continue;
        }
        superposition(c, g);
        superposition(g, c);
      }
    }
    result = complete ? SZS.Satisfiable : SZS.GaveUp;
  }

  // For each positive equation in c (both directions)
  private void superposition(Clause c, Clause d) {
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      var e = Equation.of(c.literals[i]);
      superposition(c, d, i, e.left, e.right);
      superposition(c, d, i, e.right, e.left);
    }
  }

  // For each equation in d (both directions)
  private void superposition(Clause c, Clause d, int ci, Term c0, Term c1) {
    if (c0 == Term.TRUE) {
      return;
    }
    for (var i = 0; i < d.literals.length; i++) {
      var e = Equation.of(d.literals[i]);
      superposition(c, d, ci, c0, c1, i, e.left, e.right, new ArrayList<>(), e.left);
      superposition(c, d, ci, c0, c1, i, e.right, e.left, new ArrayList<>(), e.right);
    }
  }

  // Descend into subterms
  private void superposition(
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
    if (a instanceof Variable) {
      return;
    }
    superposition1(c, d, ci, c0, c1, di, d0, d1, position, a);
    for (var i = 1; i < a.size(); i++) {
      position.add(i);
      superposition(c, d, ci, c0, c1, di, d0, d1, position, a.get(i));
      position.remove(position.size() - 1);
    }
  }

  // Check this subterm, substitute and make new clause
  private void superposition1(
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
    var map = new HashMap<Variable, Term>();
    if (!c0.unify(a, map)) {
      return;
    }
    if (lpo.greater(c1.replaceVars(map), c0.replaceVars(map))) {
      return;
    }
    var e = Equation.of(d0.splice(position, c1), d1);

    // Negative literals
    var negative = new ArrayList<Term>(c.negativeSize + d.negativeSize);
    for (var i = 0; i < c.negativeSize; i++) {
      negative.add(c.literals[i].replaceVars(map));
    }
    for (var i = 0; i < d.negativeSize; i++) {
      if (i != di) {
        negative.add(d.literals[i].replaceVars(map));
      }
    }

    // Positive literals
    var positive = new ArrayList<Term>(c.positiveSize() + d.positiveSize() - 1);
    for (var i = c.negativeSize; i < c.literals.length; i++) {
      if (i != ci) {
        positive.add(c.literals[i].replaceVars(map));
      }
    }
    for (var i = d.negativeSize; i < d.literals.length; i++) {
      if (i != di) {
        positive.add(d.literals[i].replaceVars(map));
      }
    }

    // Negative and positive superposition
    ((di < d.negativeSize) ? negative : positive).add(e.replaceVars(map).term());

    // Make new clause
    generatedSuperposition++;
    clause(new ClauseSuperposition(negative, positive, original(c), original(d)));
  }

  private static int volume(Clause c) {
    int n = c.literals.length * 16;
    for (var a : c.literals) {
      n += volume(a);
    }
    return n;
  }

  private static int volume(Term a) {
    int n = 1;
    for (var b : a) {
      n += volume(b);
    }
    return n;
  }

  private static final class ClauseFactoring extends ClauseFrom {
    ClauseFactoring(List<Term> negative, List<Term> positive, Clause from) {
      super(negative, positive, from);
    }

    @Override
    public String inference() {
      return "factoring";
    }
  }

  private static final class ClauseRenamed extends ClauseFrom {
    ClauseRenamed(Term[] literals, int negativeSize, Clause from) {
      super(literals, negativeSize, from);
    }

    @Override
    public String inference() {
      throw new UnsupportedOperationException();
    }
  }

  private static final class ClauseResolution extends ClauseFrom {
    ClauseResolution(List<Term> negative, List<Term> positive, Clause from) {
      super(negative, positive, from);
    }

    @Override
    public String inference() {
      return "resolution";
    }
  }

  private static final class ClauseSuperposition extends Clause {
    private final Clause from, from1;

    ClauseSuperposition(List<Term> negative, List<Term> positive, Clause from, Clause from1) {
      super(negative, positive);
      this.from = from;
      this.from1 = from1;
    }

    @Override
    public Formula[] from() {
      return new Formula[] {from, from1};
    }

    @Override
    public String inference() {
      return "superposition";
    }
  }
}
