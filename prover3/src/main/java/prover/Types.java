package prover;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public final class Types {
  private Types() {}

  public static Object typeof(Object a) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case EQUALS:
          case AND:
          case OR:
          case EQV:
          case NOT:
          case IS_INTEGER:
          case IS_RATIONAL:
          case EXISTS:
          case ALL:
          case LESS:
          case LESS_EQ:
            return Symbol.BOOLEAN;
          case TO_INTEGER:
            return Symbol.INTEGER;
          case TO_REAL:
            return Symbol.REAL;
          case TO_RATIONAL:
            return Symbol.RATIONAL;
          default:
            return typeof(a1.get(1));
        }
      var opType = typeof(op);
      if (!(opType instanceof List)) throw new IllegalArgumentException(a.toString());
      return ((List) opType).get(0);
    }
    if (a instanceof Func) return ((Func) a).type;
    if (a instanceof Variable) return ((Variable) a).type;
    if (a instanceof Boolean) return Symbol.BOOLEAN;
    if (a instanceof BigInteger) return Symbol.INTEGER;
    if (a instanceof BigRational) return Symbol.RATIONAL;
    if (a instanceof String) return Symbol.INDIVIDUAL;
    throw new IllegalArgumentException(a.toString());
  }

  // First step of type inference:
  // Unify to figure out how all the unspecified types can be made consistent
  private static void unifyTypes(Object wanted, Object a, HashMap<Variable, Object> map) {
    if (!Unification.unify(wanted, typeof(a), map))
      throw new IllegalArgumentException(String.format("%s != %s %s", wanted, typeof(a), a));
    if (!(a instanceof List)) return;
    var a1 = (List) a;
    var op = a1.get(0);
    if (op instanceof Symbol)
      switch ((Symbol) op) {
        case ALL:
        case EXISTS:
          unifyTypes(Symbol.BOOLEAN, a1.get(2), map);
          return;
        case AND:
        case OR:
        case EQV:
        case NOT:
          for (var i = 1; i < a1.size(); i++) unifyTypes(Symbol.BOOLEAN, a1.get(i), map);
          return;
        default:
          {
            var type = typeof(a1.get(1));
            for (var i = 1; i < a1.size(); i++) unifyTypes(type, a1.get(i), map);
            return;
          }
      }
    var opType = typeof(op);
    if (opType instanceof List) {
      var opType1 = (List) opType;
      if (opType1.size() == a1.size()) {
        for (var i = 1; i < opType1.size(); i++) unifyTypes(opType1.get(i), a1.get(i), map);
        return;
      }
    }
    throw new IllegalArgumentException(
        String.format("%s: %s %s: %s", wanted, typeof(a), a, opType));
  }

  // Second step of type inference:
  // Fill in actual types for all the type variables
  @SuppressWarnings("unchecked")
  private static void setTypes(Object a, HashMap<Variable, Object> map) {
    Etc.treeWalk(
        a,
        b -> {
          if (b instanceof Func) {
            var b1 = (Func) b;
            b1.type = Unification.replace(b1.type, map);
            if (b1.type instanceof List) {
              var type = (List) b1.type;
              b1.type = Etc.map(type, t -> t instanceof Variable ? Symbol.INDIVIDUAL : t);
              return;
            }
            if (b1.type instanceof Variable) {
              b1.type = Symbol.INDIVIDUAL;
              return;
            }
          }
        });
  }

  // Third step of type inference:
  // Check the types are correct
  private static void checkTypes(Object wanted, Object a) {
    if (wanted instanceof Variable)
      throw new IllegalArgumentException(String.format("%s: %s", wanted, a));
    if (!wanted.equals(typeof(a)))
      throw new IllegalArgumentException(String.format("%s != %s %s", wanted, typeof(a), a));
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
          case EXISTS:
            {
              var binding = (List) a1.get(1);
              for (var x : binding)
                if (typeof(x) == Symbol.BOOLEAN)
                  throw new IllegalArgumentException(String.format("%s: %s", wanted, a));
              checkTypes(Symbol.BOOLEAN, a1.get(2));
              return;
            }
          case DIVIDE_FLOOR:
          case DIVIDE_TRUNCATE:
          case DIVIDE_EUCLIDEAN:
          case REMAINDER_FLOOR:
          case REMAINDER_TRUNCATE:
          case REMAINDER_EUCLIDEAN:
            for (var i = 1; i < a1.size(); i++) checkTypes(Symbol.INTEGER, a1.get(i));
            return;
          case AND:
          case OR:
          case EQV:
          case NOT:
            for (var i = 1; i < a1.size(); i++) checkTypes(Symbol.BOOLEAN, a1.get(i));
            return;
          case EQUALS:
            {
              var type = typeof(a1.get(1));
              for (var i = 1; i < a1.size(); i++) checkTypes(type, a1.get(i));
              return;
            }
          case DIVIDE:
            {
              var type = typeof(a1.get(1));
              if (type == Symbol.INTEGER)
                throw new IllegalArgumentException(String.format("%s: %s", wanted, a));
              for (var i = 1; i < a1.size(); i++) checkTypes(type, a1.get(i));
              return;
            }
          default:
            {
              var type = typeof(a1.get(1));
              if (!isNumeric(type))
                throw new IllegalArgumentException(String.format("%s: %s", wanted, a));
              for (var i = 1; i < a1.size(); i++) checkTypes(type, a1.get(i));
              return;
            }
        }
      var opType = typeof(op);
      if (opType instanceof List) {
        var opType1 = (List) opType;
        if (opType1.size() == a1.size()) {
          for (var i = 1; i < opType1.size(); i++) checkTypes(opType1.get(i), a1.get(i));
          return;
        }
      }
      throw new IllegalArgumentException(
          String.format("%s: %s %s: %s", wanted, typeof(a), a, opType));
    }
    if (a instanceof Func) {
      var type = typeof(a);
      if (type instanceof List) {
        var type1 = (List) type;
        for (var i = 1; i < type1.size(); i++)
          if (type1.get(i) == Symbol.BOOLEAN)
            throw new IllegalArgumentException(
                String.format("%s: %s %s: %s", wanted, typeof(a), a, type));
      }
      return;
    }
    if (a instanceof Variable) {
      if (typeof(a) == Symbol.BOOLEAN)
        throw new IllegalArgumentException(String.format("%s: %s", wanted, a));
      return;
    }
  }

  public static void inferTypes(ArrayList<Formula> formulas, ArrayList<Clause> clauses) {
    var terms = new ArrayList<>();
    for (var formula : formulas) terms.add(formula.term());
    for (var c : clauses) terms.add(c.term());
    var map = new HashMap<Variable, Object>();
    for (var a : terms) unifyTypes(Symbol.BOOLEAN, a, map);
    for (var a : terms) setTypes(a, map);
    for (var a : terms) checkTypes(Symbol.BOOLEAN, a);
  }

  public static boolean isNumeric(Object a) {
    var type = typeof(a);
    if (type instanceof Symbol)
      switch ((Symbol) type) {
        case INTEGER:
        case RATIONAL:
        case REAL:
          return true;
      }
    return false;
  }
}
