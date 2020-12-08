package prover;

import java.util.*;

// Open problem:
// https://stackoverflow.com/questions/53718986/converting-first-order-logic-to-cnf-without-exponential-blowup
public final class CNF {
  private final ArrayList<Term> negative = new ArrayList<>();
  private final ArrayList<Term> positive = new ArrayList<>();
  private final List<Clause> clauses;

  // Statistics
  public Histogram histogram =
      new Histogram() {
        @Override
        public String keyHeader() {
          return "Factor";
        }

        @Override
        public void sort(Object[] keys) {
          Arrays.sort(keys, Comparator.comparingInt(o -> (int) o));
        }

        @Override
        public String valueHeader() {
          return "Count";
        }
      };
  public long timeEnd;

  public CNF(List<Formula> formulas, List<Clause> clauses) {
    this.clauses = clauses;
    for (var formula : formulas) {
      var old = clauses.size();
      convert(formula);
      histogram.inc(clauses.size() - old);
    }
    timeEnd = System.currentTimeMillis();
  }

  private static Term andGet(Term a, int i) {
    assert i > 0;
    if (a.op() == Op.AND) {
      return a.get(i);
    }
    assert i == 1;
    return a;
  }

  private static int andSize(Term a) {
    if (a.op() == Op.AND) {
      return a.size() - 1;
    }
    return 1;
  }

  private void convert(Formula formula) {
    assert !(formula instanceof Clause);

    // Variables must be bound only for the first step
    var a = formula.term();
    assert a.freeVars().isEmpty();

    // Negation normal form includes several transformations that need to be done together
    var b = nnf(new LinkedHashMap<>(), new HashMap<>(), true, a);
    a = a.unquantify();
    if (!a.isomorphic(b, new HashMap<>())) {
      formula =
          new FormulaTermFrom(b, formula) {
            @Override
            public String inference() {
              return "nnf";
            }
          };
      formula.setId();
      a = b;
    }

    // Distribute OR down into AND
    b = distribute(a);
    if (!a.equals(b)) {
      formula =
          new FormulaTermFrom(b, formula) {
            @Override
            public String inference() {
              return "distribute";
            }
          };
      formula.setId();
      a = b;
    }

    // Split AND into clauses
    var from = formula;
    for (var i = 1; i <= andSize(a); i++) {
      negative.clear();
      positive.clear();
      split(andGet(a, i));
      var c =
          new ClauseFrom(negative, positive, from) {
            @Override
            public String inference() {
              return "split";
            }
          };
      if (!c.isTrue()) {
        c.setId();
        clauses.add(c);
      }
    }
  }

  private Term distribute(Term a) {
    switch (a.op()) {
      case AND:

        // Flat layer of AND
        var r = new ArrayList<Term>();
        r.add(Term.AND);
        for (var i = 1; i < a.size(); i++) {
          var x = distribute(a.get(i));
          if (x.op() == Op.AND) {
            for (var j = 1; j < x.size(); j++) {
              r.add(x.get(j));
            }
            continue;
          }
          r.add(x);
        }
        return Term.of(r);
      case OR:

        // Flat layer of ANDs
        var ands = new ArrayList<List<Term>>(a.size());
        long total = 1;
        for (var x : a.cdr()) {
          x = distribute(x);
          if (x.op() != Op.AND) {
            ands.add(Collections.singletonList(x));
            continue;
          }
          var n = x.size() - 1;
          if ((total > 1) && (n > 1) && (total * n > 4)) {
            ands.add(Collections.singletonList(rename(x)));
            continue;
          }
          var and = new ArrayList<Term>(n);
          for (var y : x.cdr()) {
            and.add(y);
          }
          ands.add(and);
          total *= n;
        }

        // Cartesian product of ANDs
        var ors = Util.cartesianProduct(ands);
        var and = new ArrayList<Term>();
        and.add(Term.AND);
        for (var or : ors) {
          and.add(Term.of(Term.OR, or));
        }
        return Term.of(and);
    }
    return a;
  }

  private Term nnf(
      Map<Variable, Variable> all, Map<Variable, Term> exists, boolean polarity, Term a) {
    switch (a.tag()) {
      case CONST_FALSE:
        return Term.of(!polarity);
      case CONST_TRUE:
        return Term.of(polarity);
      case VAR:
        {
          var a1 = (Variable) a;
          Term b = all.get(a1);
          if (b != null) {
            return b;
          }
          b = exists.get(a1);
          assert b != null;
          return b;
        }
    }
    switch (a.op()) {
      case ALL:
        {
          return polarity ? nnfAll(all, exists, true, a) : nnfExists(all, exists, false, a);
        }
      case AND:
        {
          var r = new Term[a.size()];
          r[0] = polarity ? Term.AND : Term.OR;
          for (var i = 1; i < a.size(); i++) {
            r[i] = nnf(all, exists, polarity, a.get(i));
          }
          return Term.of(r);
        }
      case EQV:
        {
          var x = a.get(1);
          var y = a.get(2);
          var z1 = nnf(all, exists, false, x).or(nnf(all, exists, polarity, y));
          var z2 = nnf(all, exists, true, x).or(nnf(all, exists, !polarity, y));
          return z1.and(z2);
        }
      case EXISTS:
        {
          return polarity ? nnfExists(all, exists, true, a) : nnfAll(all, exists, false, a);
        }
      case NOT:
        return nnf(all, exists, !polarity, a.get(1));
      case OR:
        {
          var r = new Term[a.size()];
          r[0] = polarity ? Term.OR : Term.AND;
          for (var i = 1; i < a.size(); i++) {
            r[i] = nnf(all, exists, polarity, a.get(i));
          }
          return Term.of(r);
        }
    }
    a = a.map1(b -> nnf(all, exists, true, b));
    return polarity ? a : a.not();
  }

  private Term nnfAll(
      Map<Variable, Variable> all, Map<Variable, Term> exists, boolean polarity, Term a) {
    var all1 = new LinkedHashMap<>(all);
    for (var x : a.get(1)) {
      all1.put((Variable) x, new Variable(x.type()));
    }
    return nnf(all1, exists, polarity, a.get(2));
  }

  private Term nnfExists(
      Map<Variable, Variable> all, Map<Variable, Term> exists, boolean polarity, Term a) {
    var exists1 = new HashMap<>(exists);
    for (var x : a.get(1)) {
      exists1.put((Variable) x, skolem(x.type(), all.values()));
    }
    return nnf(all, exists1, polarity, a.get(2));
  }

  private Term rename(Term a) {
    var b = skolem(Type.BOOLEAN, a.freeVars());
    var formula = new FormulaTerm(b.implies(a), null);
    formula.setId();
    convert(formula);
    return b;
  }

  private static Term skolem(Type type, Collection<Variable> args) {
    if (args.isEmpty()) {
      return new Function(type, null);
    }
    return new Function(type, null, args).call(args);
  }

  private void split(Term a) {
    switch (a.op()) {
      case AND:
        throw new IllegalArgumentException(a.toString());
      case NOT:
        negative.add(a.get(1));
        break;
      case OR:
        for (var b : a.cdr()) {
          split(b);
        }
        break;
      default:
        positive.add(a);
        break;
    }
  }
}
