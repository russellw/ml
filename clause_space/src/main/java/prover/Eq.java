package prover;

import java.util.Map;
import java.util.function.Function;

public final class Eq extends Term2 {
  public Eq(Term left, Term right) {
    super(left, right);
    if (!equatable(left, right)) {
      throw new IllegalArgumentException(toString());
    }
  }

  public static boolean equatable(Term a, Term b) {
    var type = a.type();
    if (type != b.type()) {
      return false;
    }
    return (type != Type.BOOLEAN) || (b == Term.TRUE);
  }

  @Override
  public Term eval(Map<Variable, Term> map) {
    return of(get(0).eval(map).equals(get(1).eval(map)));
  }

  public static Eq of(Term a) {
    if (a.type() != Type.BOOLEAN) {
      throw new IllegalArgumentException(a + ": " + a.type());
    }
    if (a instanceof Eq) {
      return (Eq) a;
    }
    return new Eq(a, Term.TRUE);
  }

  public Eq replace(Map<Variable, Term> map) {
    return new Eq(left.replace(map), right.replace(map));
  }

  @Override
  public Term simplify() {
    if (left.equals(right)) {
      return TRUE;
    }
    return this;
  }

  @Override
  public Tag tag() {
    return Tag.EQ;
  }

  public Term term() {
    if (right == Term.TRUE) {
      return left;
    }
    return this;
  }

  @Override
  public Term transform(Function<Term, Term> f) {
    return new Eq(f.apply(left), f.apply(right));
  }

  @Override
  public Type type() {
    return Type.BOOLEAN;
  }
}
