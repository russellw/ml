package prover;

import java.math.BigInteger;
import java.util.*;

public final class Terms {
  private abstract static class Op1 {
    abstract Object apply(BigInteger x);

    abstract Object apply(BigRational x);
  }

  private abstract static class Op2 {
    abstract Object apply(BigInteger x, BigInteger y);

    abstract Object apply(BigRational x, BigRational y);
  }

  private static Object eval1(Object a, Object x, Op1 op) {
    if (x instanceof BigInteger) {
      var x1 = (BigInteger) x;
      return op.apply(x1);
    }
    if (x instanceof BigRational) {
      var x1 = (BigRational) x;
      return op.apply(x1);
    }
    if (x instanceof List) {
      var x1 = (List) x;
      if (x1.get(0) == Symbol.TO_REAL) {
        var x2 = x1.get(1);
        if (x2 instanceof BigRational) {
          var x3 = (BigRational) x2;
          var r = op.apply(x3);
          if (r instanceof BigRational) return List.of(Symbol.TO_REAL, r);
          return r;
        }
      }
    }
    return a;
  }

  private static Object eval2(Object a, Object x, Object y, Op2 op) {
    if (x instanceof BigInteger) {
      var x1 = (BigInteger) x;
      if (y instanceof BigInteger) {
        var y1 = (BigInteger) y;
        return op.apply(x1, y1);
      }
      return a;
    }
    if (x instanceof BigRational) {
      var x1 = (BigRational) x;
      if (y instanceof BigRational) {
        var y1 = (BigRational) y;
        return op.apply(x1, y1);
      }
      return a;
    }
    if (x instanceof List) {
      var x1 = (List) x;
      if (x1.get(0) == Symbol.TO_REAL) {
        var x2 = x1.get(1);
        if (x2 instanceof BigRational) {
          var x3 = (BigRational) x2;
          if (y instanceof List) {
            var y1 = (List) y;
            if (y1.get(0) == Symbol.TO_REAL) {
              var y2 = y1.get(1);
              if (y2 instanceof BigRational) {
                var y3 = (BigRational) y2;
                var r = op.apply(x3, y3);
                if (r instanceof BigRational) return List.of(Symbol.TO_REAL, r);
                return r;
              }
            }
          }
        }
      }
    }
    return a;
  }

  private Terms() {}

  @SuppressWarnings("unchecked")
  public static Object simplify(Object a0) {
    if (!(a0 instanceof List)) return a0;
    var a = (List) a0;
    a = Etc.map(a, Terms::simplify);
    var op = a.get(0);
    if (!(op instanceof Symbol)) return a;
    var op1 = (Symbol) op;
    var x = a.get(1);
    Object y = null;
    if (a.size() > 2) y = a.get(2);
    switch (op1) {
      case EQUALS:
        if (x.equals(y)) return true;
        if (constant(x) && constant(y)) return false;
        break;
      case TO_INTEGER:
        if (Types.typeof(x) == Symbol.INTEGER) return x;
        if (x instanceof BigRational) {
          var x1 = (BigRational) x;
          return x1.floor();
        }
        if (x instanceof List) {
          var x1 = (List) x;
          if (x1.get(0) == Symbol.TO_REAL) {
            var x2 = x1.get(1);
            if (x2 instanceof BigRational) {
              var x3 = (BigRational) x2;
              return x3.floor();
            }
          }
        }
        return a;
      case TO_RATIONAL:
        if (x instanceof BigInteger) {
          var x1 = (BigInteger) x;
          return BigRational.of(x1);
        }
        if (Types.typeof(x) == Symbol.RATIONAL) return x;
        if (x instanceof List) {
          var x1 = (List) x;
          if (x1.get(0) == Symbol.TO_REAL) {
            var x2 = x1.get(1);
            if (Types.typeof(x2) == Symbol.RATIONAL) return x2;
          }
        }
        return a;
      case TO_REAL:
        if (Types.typeof(x) == Symbol.REAL) return x;
        return a;
      case IS_INTEGER:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return true;
              }

              @Override
              Object apply(BigRational x) {
                return x.den.equals(BigInteger.ONE);
              }
            });
      case IS_RATIONAL:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return true;
              }

              @Override
              Object apply(BigRational x) {
                return true;
              }
            });
      case NEGATE:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return x.negate();
              }

              @Override
              Object apply(BigRational x) {
                return x.negate();
              }
            });
      case CEIL:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return x;
              }

              @Override
              Object apply(BigRational x) {
                return BigRational.of(x.ceil());
              }
            });
      case FLOOR:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return x;
              }

              @Override
              Object apply(BigRational x) {
                return BigRational.of(x.floor());
              }
            });
      case ROUND:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return x;
              }

              @Override
              Object apply(BigRational x) {
                return BigRational.of(x.round());
              }
            });
      case TRUNCATE:
        return eval1(
            a,
            x,
            new Op1() {
              @Override
              Object apply(BigInteger x) {
                return x;
              }

              @Override
              Object apply(BigRational x) {
                return BigRational.of(x.truncate());
              }
            });
      case ADD:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.add(y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.add(y);
              }
            });
      case SUBTRACT:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.subtract(y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.subtract(y);
              }
            });
      case MULTIPLY:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.multiply(y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.multiply(y);
              }
            });
      case DIVIDE:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                throw new IllegalArgumentException(a0.toString());
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.divide(y);
              }
            });
      case DIVIDE_EUCLIDEAN:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return Etc.divideEuclidean(x, y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.divideEuclidean(y);
              }
            });
      case DIVIDE_FLOOR:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return Etc.divideFloor(x, y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.divideFloor(y);
              }
            });
      case DIVIDE_TRUNCATE:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.divide(y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.divideTruncate(y);
              }
            });
      case REMAINDER_EUCLIDEAN:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return Etc.remainderEuclidean(x, y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.remainderEuclidean(y);
              }
            });
      case REMAINDER_FLOOR:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return Etc.remainderFloor(x, y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.remainderFloor(y);
              }
            });
      case REMAINDER_TRUNCATE:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.remainder(y);
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.remainderTruncate(y);
              }
            });
      case LESS:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.compareTo(y) < 0;
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.compareTo(y) < 0;
              }
            });
      case LESS_EQ:
        return eval2(
            a,
            x,
            y,
            new Op2() {
              @Override
              Object apply(BigInteger x, BigInteger y) {
                return x.compareTo(y) <= 0;
              }

              @Override
              Object apply(BigRational x, BigRational y) {
                return x.compareTo(y) <= 0;
              }
            });
    }
    return a;
  }

  public static boolean match(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return false;

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

  public static boolean occurs(Variable a, Object b, Map<Variable, Object> map) {
    if (b instanceof Variable) {
      if (a == b) return true;
      var b1 = map.get(b);
      if (b1 != null) return occurs(a, b1, map);
      return false;
    }
    if (b instanceof List) {
      var b1 = (List) b;
      for (var x : b1) if (occurs(a, x, map)) return true;
    }
    return false;
  }

  private static boolean unifyVariable(Variable a, Object b, Map<Variable, Object> map) {
    // Existing mappings
    var a1 = map.get(a);
    if (a1 != null) return unify(a1, b, map);
    if (b instanceof Variable) {
      var b1 = map.get(b);
      if (b1 != null) return unify(a, b1, map);
    }

    // Occurs check
    if (occurs(a, b, map)) return false;

    // New mapping
    map.put(a, b);
    return true;
  }

  @SuppressWarnings("unchecked")
  public static boolean constant(Object a) {
    if (a instanceof List) return Etc.all((List) a, Terms::constant);
    return !(a instanceof Func || a instanceof Variable);
  }

  public static boolean unify(Object a, Object b, Map<Variable, Object> map) {
    // Equal
    if (a == b) return true;

    // Type mismatch
    if (!Types.typeof(a).equals(Types.typeof(b))) return false;

    // Variable
    if (a instanceof Variable) return unifyVariable((Variable) a, b, map);
    if (b instanceof Variable) return unifyVariable((Variable) b, a, map);

    // Compounds
    if (a instanceof List) {
      var a1 = (List) a;
      if (b instanceof List) {
        var b1 = (List) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        if (a1.get(0) != b1.get(0)) return false;
        for (var i = 1; i < n; i++) if (!unify(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

  public static Object replace(Object a, Map<Variable, Object> map) {
    return Etc.mapLeaves(
        a,
        b -> {
          if (b instanceof Variable) {
            var b1 = map.get(b);
            if (b1 != null) return replace(b1, map);
          }
          return b;
        });
  }

  @SuppressWarnings("unchecked")
  private static void getFreeVariables(Set<Variable> bound, Object a, Set<Variable> r) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
          case EXISTS:
            {
              var binding = (List) a1.get(1);
              bound = new HashSet<>(bound);
              bound.addAll(binding);
              getFreeVariables(bound, a1.get(2), r);
              return;
            }
        }
      for (var b : a1) getFreeVariables(bound, b, r);
      return;
    }
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      if (!bound.contains(a1)) r.add(a1);
    }
  }

  public static Set<Variable> freeVariables(Object a) {
    var r = new LinkedHashSet<Variable>();
    getFreeVariables(new HashSet<>(), a, r);
    return r;
  }

  public static Object quantify(Object a) {
    var variables = freeVariables(a);
    if (variables.isEmpty()) return a;
    return List.of(Symbol.ALL, List.copyOf(variables), a);
  }

  public static Object unquantify(Object a) {
    while (a instanceof List) {
      var a1 = (List) a;
      if (a1.get(0) != Symbol.ALL) break;
      a = a1.get(2);
    }
    return a;
  }

  public static boolean isomorphic(Object a, Object b, Map<Variable, Variable> map) {
    // Equal
    if (a == b) return true;

    // Variable
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      if (b instanceof Variable) {
        var b1 = (Variable) b;
        var a2 = map.get(a1);
        var b2 = map.get(b1);

        // Compatible mapping
        if (a1 == b2 && b1 == a2) return true;

        // New mapping
        if (a2 == null && b2 == null) {
          map.put(a1, b1);
          map.put(b1, a1);
          return true;
        }
      }
      return false;
    }

    // Compounds
    if (a instanceof List) {
      var a1 = (List) a;
      if (b instanceof List) {
        var b1 = (List) b;
        int n = a1.size();
        if (n != b1.size()) return false;
        for (var i = 0; i < n; i++) if (!isomorphic(a1.get(i), b1.get(i), map)) return false;
        return true;
      }
      return false;
    }

    // Atoms
    return a.equals(b);
  }

  public static List<Object> implies(Object a, Object b) {
    return List.of(Symbol.OR, List.of(Symbol.NOT, a), b);
  }
}
