package prover;

import java.util.*;
import java.util.regex.Pattern;

// Open problem:
// https://stackoverflow.com/questions/53718986/converting-first-order-logic-to-cnf-without-exponential-blowup
public final class CNF {
  private static final Pattern SKOLEM_PATTERN = Pattern.compile("\\s*Status\\s*:\\s*(\\w+)");
  private final Problem problem;
  private int skolemId = -1;
  private final List<Object> negative = new ArrayList<>();
  private final List<Object> positive = new ArrayList<>();

  private Func skolem(Object type) {
    var a = new Func(type, "sK" + ++skolemId);
    problem.skolems.add(a);
    return a;
  }

  private Object skolem(Object returnType, Collection<Variable> args) {
    // Atom
    if (args.isEmpty()) return skolem(returnType);

    // Type
    var type = new ArrayList<>();
    type.add(returnType);
    for (var a : args) type.add(Types.typeof(a));

    // Term
    var r = new ArrayList<>();
    r.add(skolem(type));
    r.addAll(args);
    return r;
  }

  private Object nnfAll(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, List a) {
    var binding = (List) a.get(1);
    all = new LinkedHashMap<>(all);
    for (var x : binding) all.put((Variable) x, new Variable(Types.typeof(x)));
    return nnf(all, exists, polarity, a.get(2));
  }

  private Object nnfExists(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, List a) {
    var binding = (List) a.get(1);
    exists = new HashMap<>(exists);
    for (var x : binding) exists.put((Variable) x, skolem(Types.typeof(x), all.values()));
    return nnf(all, exists, polarity, a.get(2));
  }

  @SuppressWarnings("unchecked")
  private Object nnf(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
            return polarity ? nnfAll(all, exists, true, a1) : nnfExists(all, exists, false, a1);
          case AND:
            {
              var r = new Object[a1.size()];
              r[0] = polarity ? Symbol.AND : Symbol.OR;
              for (var i = 1; i < a1.size(); i++) r[i] = nnf(all, exists, polarity, a1.get(i));
              return List.of(r);
            }
          case EQV:
            {
              var x = a1.get(1);
              var y = a1.get(2);
              return List.of(
                  Symbol.AND,
                  List.of(Symbol.OR, nnf(all, exists, false, x), nnf(all, exists, polarity, y)),
                  List.of(Symbol.OR, nnf(all, exists, true, x), nnf(all, exists, !polarity, y)));
            }
          case EXISTS:
            return polarity ? nnfExists(all, exists, true, a1) : nnfAll(all, exists, false, a1);
          case NOT:
            return nnf(all, exists, !polarity, a1.get(1));
          case OR:
            {
              var r = new Object[a1.size()];
              r[0] = polarity ? Symbol.OR : Symbol.AND;
              for (var i = 1; i < a1.size(); i++) r[i] = nnf(all, exists, polarity, a1.get(i));
              return List.of(r);
            }
        }
      a = Etc.map(a1, b -> nnf(all, exists, true, b));
    } else if (a instanceof Variable) {
      var a1 = (Variable) a;
      Object b = all.get(a1);
      if (b != null) return b;
      b = exists.get(a1);
      assert b != null;
      return b;
    } else if (a instanceof Boolean) return polarity == (boolean) a;
    return polarity ? a : List.of(Symbol.NOT, a);
  }

  private Object rename(Object a) {
    var b = skolem(Symbol.BOOLEAN, Terms.freeVariables(a));
    var formula = new Formula(Terms.quantify(Terms.implies(b, a)), Inference.DEFINE);
    convert(formula);
    return b;
  }

  private Object distribute(Object a) {
    if (!(a instanceof List)) return a;
    var a1 = (List) a;
    var op = a1.get(0);
    if (!(op instanceof Symbol)) return a;
    switch ((Symbol) op) {
      case AND:
        {
          // Flat layer of AND
          var r = new ArrayList<>();
          r.add(Symbol.AND);
          for (var i = 1; i < a1.size(); i++) {
            var b = distribute(a1.get(i));
            if (b instanceof List) {
              var b1 = (List) b;
              if (b1.get(0) == Symbol.AND) {
                for (var j = 1; j < b1.size(); j++) r.add(b1.get(j));
                continue;
              }
            }
            r.add(b);
          }
          return r;
        }
      case OR:
        {
          // Flat layer of ANDs
          var ands = new ArrayList<List<Object>>(a1.size());
          long total = 1;
          for (var i = 1; i < a1.size(); i++) {
            var b = distribute(a1.get(i));
            if (b instanceof List) {
              var b1 = (List) b;
              if (b1.get(0) != Symbol.AND) {
                ands.add(Collections.singletonList(b));
                continue;
              }
              var n = b1.size() - 1;
              if (total > 1 && n > 1 && total * n > 4) {
                ands.add(Collections.singletonList(rename(b)));
                continue;
              }
              var and = new ArrayList<>(n);
              for (var j = 1; j < b1.size(); j++) and.add(b1.get(j));
              ands.add(and);
              total *= n;
              continue;
            }
            ands.add(Collections.singletonList(b));
          }

          // Cartesian product of ANDs
          var ors = Etc.cartesianProduct(ands);
          var and = new ArrayList<>();
          and.add(Symbol.AND);
          for (var or : ors) {
            or.add(0, Symbol.OR);
            and.add(or);
          }
          return and;
        }
    }
    return a1;
  }

  private void split(Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case AND:
            throw new IllegalArgumentException(a.toString());
          case NOT:
            negative.add(a1.get(1));
            return;
          case OR:
            for (var i = 1; i < a1.size(); i++) split(a1.get(i));
            return;
        }
    }
    positive.add(a);
  }

  private void convert(Formula formula) {
    // Variables must be bound only for the first step
    var a = formula.term();
    assert Terms.freeVariables(a).isEmpty();

    // Negation normal form includes several transformations that need to be done together
    var b = nnf(new LinkedHashMap<>(), new HashMap<>(), true, a);
    a = Terms.unquantify(a);
    if (!Terms.isomorphic(a, b, new HashMap<>())) {
      formula = new Formula(b, Inference.NNF, formula);
      a = b;
    }

    // Distribute OR down into AND
    b = distribute(a);
    if (!a.equals(b)) {
      formula = new Formula(b, Inference.DISTRIBUTE, formula);
      a = b;
    }

    // Split AND into clauses
    if (a instanceof List) {
      var a1 = (List) a;
      if (a1.get(0) == Symbol.AND) {
        for (var i = 1; i < a1.size(); i++) {
          negative.clear();
          positive.clear();
          split(a1.get(i));
          var c = new Clause(negative, positive, Inference.SPLIT, formula);
          if (!c.isTrue()) problem.clauses.add(c);
        }
        return;
      }
    }
    negative.clear();
    positive.clear();
    split(a);
    var c = new Clause(negative, positive, Inference.SPLIT, formula);
    if (!c.isTrue()) problem.clauses.add(c);
  }

  public CNF(Problem problem) {
    this.problem = problem;
    for (var a : problem.funcs.values()) {
      var matcher = SKOLEM_PATTERN.matcher(a.name);
      if (matcher.matches()) skolemId = Math.max(skolemId, Integer.parseInt(matcher.group(1)));
    }
    for (var formula : problem.formulas) convert(formula);
  }
}
