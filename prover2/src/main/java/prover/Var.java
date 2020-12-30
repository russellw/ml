package prover;

import java.util.Map;

public final class Var extends Term {
  @Override
  public Term replace(Map<Var, Term> map) {
    var a = map.get(this);
    if (a != null) return a.replace(map);
    return this;
  }

  @Override
  public boolean isBoolean() {
    return false;
  }
}
