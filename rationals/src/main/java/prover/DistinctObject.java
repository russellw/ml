package prover;

import java.util.Map;

public final class DistinctObject extends Term {
  private final String name;

  public DistinctObject(String name) {
    this.name = name;
  }

  @Override
  public Term eval(Map<Variable, Term> map) {
    return this;
  }

  @Override
  public Tag tag() {
    return Tag.DISTINCT_OBJECT;
  }

  @Override
  public String toString() {
    return Util.quote('"', name);
  }

  @Override
  public Type type() {
    return Type.INDIVIDUAL;
  }
}
