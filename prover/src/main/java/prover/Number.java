package prover;

import java.util.Objects;

public final class Number extends Term {
  private final double value;

  public Number(double value) {
    this.value = value;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if ((o == null) || (getClass() != o.getClass())) {
      return false;
    }
    Number b = (Number) o;
    return value == b.value;
  }

  @Override
  public int hashCode() {
    return Objects.hash(value);
  }

  @Override
  public boolean isConstant() {
    return true;
  }

  @Override
  public double number() {
    return value;
  }

  @Override
  public Tag tag() {
    return Tag.NUMBER;
  }

  @Override
  public String toString() {
    return Double.toString(value);
  }

  @Override
  public Type type() {
    return Type.NUMBER;
  }
}
