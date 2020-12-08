package prover;

import java.util.Map;
import java.util.Objects;

public final class Equation {
  public final Term left, right;

  public Equation(Term left, Term right) {
    if (!equatable(left, right)) {
      throw new IllegalArgumentException(left + " = " + right);
    }
    this.left = left;
    this.right = right;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if ((o == null) || (getClass() != o.getClass())) {
      return false;
    }
    Equation o1 = (Equation) o;
    return Objects.equals(left, o1.left) && Objects.equals(right, o1.right);
  }

  public static boolean equatable(Term a, Term b) {
    if (!a.type().equals(b.type())) {
      return false;
    }
    return (b.type() != Type.BOOLEAN) || (b == Term.TRUE);
  }

  @Override
  public int hashCode() {
    return Objects.hash(left, right);
  }

  public static Equation of(Term a) {
    if (!a.type().equals(Type.BOOLEAN)) {
      throw new IllegalArgumentException(a.type().toString() + ' ' + a);
    }
    if (a.op() == Op.EQ) {
      return of(a.get(1), a.get(2));
    }
    return of(a, Term.TRUE);
  }

  public static Equation of(Term left, Term right) {
    return new Equation(left, right);
  }

  public Equation replaceVars(Map<Variable, Term> map) {
    return of(left.replaceVars(map), right.replaceVars(map));
  }

  public Term term() {
    if (right == Term.TRUE) {
      return left;
    }
    return left.eq(right);
  }

  @Override
  public String toString() {
    return left.toString() + '=' + right;
  }
}
