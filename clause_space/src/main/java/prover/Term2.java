package prover;

import java.util.Objects;

public abstract class Term2 extends Term {
  public final Term left, right;

  public Term2(Term left, Term right) {
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
    Term2 b = (Term2) o;
    return Objects.equals(left, b.left) && Objects.equals(right, b.right);
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
  public int hashCode() {
    return Objects.hash(left, right);
  }

  @Override
  public int size() {
    return 2;
  }
}
