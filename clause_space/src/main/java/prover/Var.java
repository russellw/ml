package prover;

import java.util.Map;

public final class Var extends Term {
  @Override
  public Term rename(Map<Var, Var> map) {
    var a = map.get(this);
    if (a == null) {
      a = new Var();
      map.put(this, a);
    }
    return a;
  }

  @Override
  public Term replace(Map<Var, Term> map) {
    var a = map.get(this);
    if (a != null) return a.replace(map);
    return this;
  }

  @Override
  public Tag tag() {
    return Tag.VARIABLE;
  }
}
