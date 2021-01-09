package prover;

import io.vavr.collection.Seq;
import java.math.BigInteger;
import java.util.HashMap;

public final class Types {
  private Types() {}

  public static Object typeof(Object a) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
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
      return ((Seq) typeof(op)).head();
    }
    if (a instanceof Func) return ((Func) a).type;
    if (a instanceof Variable) return ((Variable) a).type;
    if (a instanceof Boolean) return Symbol.BOOLEAN;
    if (a instanceof BigInteger) return Symbol.INTEGER;
    if (a instanceof BigRational) return Symbol.RATIONAL;
    if (a instanceof String) return Symbol.INDIVIDUAL;
    throw new IllegalArgumentException(a.toString());
  }

  private static boolean occurs(Variable a, Object b, HashMap<Variable, Object> map) {
    if (b instanceof Variable) {
      if (a == b) return true;
      var b1 = map.get(b);
      if (b1 != null) return occurs(a, b1, map);
      return false;
    }
    if (b instanceof Seq) {
      var b1 = (Seq) b;
      for (var x : b1) if (occurs(a, x, map)) return true;
    }
    return false;
  }

  private static boolean unifyVariable(Variable a, Object b, HashMap<Variable, Object> map) {
    // Existing mapping
    var a1 = map.get(a);
    if (a1 != null) return unify(a1, b, map);

    // Variable
    if (b instanceof Variable) {
      var b1 = map.get(b);
      if (b1 != null) return unify(b1, a, map);
    }

    // Occurs check
    if (occurs(a, b, map)) return false;

    // New mapping
    map.put(a, b);
    return true;
  }

  // This version of unify skips the type check
  // because it makes no sense to ask the type of a type
  private static boolean unify(Object a, Object b, HashMap<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Variable
    if (a instanceof Variable) return unifyVariable((Variable) a, b, map);
    if (b instanceof Variable) return unifyVariable((Variable) b, a, map);

    // Compounds
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      if (b instanceof Seq) {
        var b1 = (Seq) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        for (var i = 0; i < n; i++) if (!unify(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return false;
  }

  // First step of type inference:
  // Unify to figure out how all the unspecified types can be made consistent
  private static void unifyTypes(Object wanted, Object a, HashMap<Variable, Object> map) {
    if (!unify(wanted, typeof(a), map))
      throw new IllegalArgumentException(String.format("%s != %s %s", wanted, typeof(a), a));
    if (!(a instanceof Seq)) return;
    var a1 = (Seq) a;
    var op = a1.head();
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
    if (opType instanceof Seq) {
      var opType1 = (Seq) opType;
      if (opType1.size() == a1.size()) {
        for (var i = 1; i < opType1.size(); i++)
          if (!opType1.get(i).equals(typeof(a1.get(i))))
            throw new IllegalArgumentException(
                String.format("%s: %s %s: %s", wanted, typeof(a), a, opType));
        return;
      }
    }
    throw new IllegalArgumentException(
        String.format("%s: %s %s: %s", wanted, typeof(a), a, opType));
  }

  // Second step of type inference:
  // Fill in actual types for all the type variables
  private static void setTypes(Object a, HashMap<Variable, Object> map) {
    if (a instanceof Seq) {
      for (var b : (Seq) a) setTypes(b, map);
      return;
    }
    if (a instanceof Func) {
      return;
    }
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
