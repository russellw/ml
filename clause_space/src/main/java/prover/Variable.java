package prover;

import java.util.Map;

public final class Variable extends Term {
  @Override
  public boolean contains(Variable x, Map<Variable, Term> map) {
    if (this == x) {
      return true;
    }
    var a = map.get(this);
    if (a != null) {
      return a.contains(x, map);
    }
    return false;
  }

  @Override
  public Term eval(Map<Variable, Term> map) {
    return map.get(this);
  }

  @Override
  public boolean match(Term b, Map<Variable, Term> map) {
    // Equal?
    if (this == b) {
      return true;
    }

    // Existing mapping
    var a2 = map.get(this);
    if (a2 != null) {
      return a2.equals(b);
    }

    // New mapping
    map.put(this, b);
    return true;
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
