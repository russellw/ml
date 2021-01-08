package prover;

import io.vavr.collection.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// Open problem:
// https://stackoverflow.com/questions/53718986/converting-first-order-logic-to-cnf-without-exponential-blowup
public final class CNF {
  private final ArrayList<Object> negative = new ArrayList<>();
  private final ArrayList<Object> positive = new ArrayList<>();
  private final ArrayList<Clause> clauses;

  private Object nnfAll(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, Seq a) {
    var binding = (Seq) a.get(1);
    for (var x : binding) all = all.put((Variable) x, new Variable(Types.typeof(x)));
    return nnf(all, exists, polarity, a.get(2));
  }

  private Object nnfExists(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, Seq a) {
    var binding = (Seq) a.get(1);
    for (var x : binding) exists = exists.put((Variable) x, skolem(Types.typeof(x), all.values()));
    return nnf(all, exists, polarity, a.get(2));
  }

  @SuppressWarnings("unchecked")
  private Object nnf(
      Map<Variable, Variable> all, Map<Variable, Object> exists, boolean polarity, Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
            return polarity ? nnfAll(all, exists, true, a1) : nnfExists(all, exists, false, a1);
          case AND:
            {
              var r = new Object[a1.size()];
              r[0] = polarity ? Symbol.AND : Symbol.OR;
              for (var i = 1; i < a1.size(); i++) r[i] = nnf(all, exists, polarity, a1.get(i));
              return Array.of(r);
            }
          case EQV:
            {
              var x = a1.get(1);
              var y = a1.get(2);
              return Array.of(
                  Symbol.AND,
                  Array.of(Symbol.OR, nnf(all, exists, false, x), nnf(all, exists, polarity, y)),
                  Array.of(Symbol.OR, nnf(all, exists, true, x), nnf(all, exists, !polarity, y)));
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
              return Array.of(r);
            }
        }
      a = a1.map(b -> nnf(all, exists, true, b));
    } else if (a instanceof Variable) {
      var a1 = (Variable) a;
      Object b = all.getOrElse(a1, null);
      if (b != null) return b;
      b = exists.get(a1);
      assert b != null;
      return b;
    } else if (a instanceof Boolean) return polarity ^ (boolean) a;
    return polarity ? a : Array.of(Symbol.NOT, a);
  }

  private Object rename(Object a) {
    var b = skolem(Symbol.BOOLEAN, Array.ofAll(Etc.freeVariables(a)));
    var formula = new Formula(Etc.implies(b, a), Inference.DEFINE);
    convert(formula);
    return b;
  }

  private Object distribute(Object a) {
    if (!(a instanceof Seq)) return a;
    var a1 = (Seq) a;
    var op = a1.head();
    if (!(op instanceof Symbol)) return a;
    switch ((Symbol) op) {
      case AND:
        {
          // Flat layer of AND
          var r = new ArrayList<>();
          r.add(Symbol.AND);
          for (var i = 1; i < a1.size(); i++) {
            var b = distribute(a1.get(i));
            if (b instanceof Seq) {
              var b1 = (Seq) b;
              if (b1.head() == Symbol.AND) {
                for (var j = 1; j < b1.size(); j++) r.add(b1.get(j));
                continue;
              }
            }
            r.add(b);
          }
          return Array.ofAll(r);
        }
      case OR:
        {
          // Flat layer of ANDs
          var ands = new ArrayList<List<Object>>(a1.size());
          long total = 1;
          for (var i = 1; i < a1.size(); i++) {
            var b = distribute(a1.get(i));
            if (b instanceof Seq) {
              var b1 = (Seq) b;
              if (b1.head() != Symbol.AND) {
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
            and.add(Array.ofAll(or));
          }
          return Array.ofAll(and);
        }
    }
    return a1;
  }

  private static Object skolem(Object returnType, Seq<Variable> args) {
    if (args.isEmpty()) return new Func(returnType, null);
    var type = new Object[1 + args.size()];
    type[0] = returnType;
    for (var i = 0; i < args.size(); i++) type[1 + i] = args.get(i);
    return new Func(Array.of(type), null).call(args);
  }

  private void split(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
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
    assert Etc.freeVariables(a).isEmpty();

    // Negation normal form includes several transformations that need to be done together
    var b = nnf(LinkedHashMap.empty(), HashMap.empty(), true, a);
    a = Etc.unquantify(a);
    if (!Etc.isomorphic(a, b, new java.util.HashMap<>())) {
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
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.head() == Symbol.AND) {
        for (var i = 1; i <= a1.size(); i++) {
          negative.clear();
          positive.clear();
          split(a1.get(i));
          var c = new Clause(negative, positive, Inference.SPLIT, formula);
          if (!c.isTrue()) clauses.add(c);
        }
        return;
      }
    }
    negative.clear();
    positive.clear();
    split(a);
    var c = new Clause(negative, positive, Inference.SPLIT, formula);
    if (!c.isTrue()) clauses.add(c);
  }

  public CNF(ArrayList<Formula> formulas, ArrayList<Clause> clauses) {
    this.clauses = clauses;
    for (var formula : formulas) convert(formula);
  }
}
