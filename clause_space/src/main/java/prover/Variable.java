package prover;

import java.util.Map;

public final class Variable extends Term {
  @Override
  public Term eval(Map<Variable, Term> map) {
    return map.get(this);
  }

  @Override
  public Term rename(Map<Variable, Variable> map) {
    var a = map.get(this);
    if (a == null) {
      a = new Variable();
      map.put(this, a);
    }
    return a;
  }

  @Override
  public Term replace(Map<Variable, Term> map) {
    var a = map.get(this);
    if (a != null) {
      return a.replace(map);
    }
    return this;
  }

  @Override
  public Tag tag() {
    return Tag.VARIABLE;
  }
}
