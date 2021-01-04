package prover;

import java.util.Map;

public final class Variable extends Term {
  @Override
  public Term replace(Map<Variable, Term> map) {
    var a = map.get(this);
    if (a != null) return a.replace(map);
    return this;
  }

  @Override
  public boolean isBoolean() {
    return false;
  }
}
