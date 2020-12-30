package prover;

import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

public final class Eq extends Term {
  public final Term left, right;

  public Eq(Term left, Term right) {
    if (!equatable(left, right)) throw new IllegalArgumentException(toString());
    this.left = left;
    this.right = right;
  }

  @Override
  public Term get(int i) {
    switch (i) {
      case 0:
        return left;
      case 1:
        return right;
    }
    throw new IllegalArgumentException(toString() + '[' + i + ']');
  }

  @Override
  public int size() {
    return 2;
  }

  @Override
  public boolean isBoolean() {
    return true;
  }

  public static boolean equatable(Term a, Term b) {
    var type = a.isBoolean();
    if (type != b.isBoolean()) return false;
    return !type || (b == Term.TRUE);
  }

  public static Eq of(Term a) {
    if (!a.isBoolean()) throw new IllegalArgumentException(a.toString());
    if (a instanceof Eq) return (Eq) a;
    return new Eq(a, Term.TRUE);
  }

  public Eq replace(Map<Var, Term> map) {
    return new Eq(left.replace(map), right.replace(map));
  }

  public Term term() {
    if (right == Term.TRUE) return left;
    return this;
  }

  @Override
  public Term transform(Function<Term, Term> f) {
    return new Eq(f.apply(left), f.apply(right));
  }
}
