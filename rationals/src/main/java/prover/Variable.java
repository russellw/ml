package prover;

import java.util.Map;
import java.util.Set;

public final class Variable extends Term {
  private Type type;
  private String name;

  public Variable(Type type) {
    this.type = type;
  }

  public Variable(Type type, String name) {
    this.type = type;
    this.name = name;
  }

  @Override
  public boolean contains(Variable x) {
    return this == x;
  }

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
  public Term freshVars(Map<Variable, Variable> map) {
    var a = map.get(this);
    if (a == null) {
      a = new Variable(type);
      map.put(this, a);
    }
    return a;
  }

  @Override
  public void getVars(Set<Variable> r) {
    r.add(this);
  }

  @Override
  public void getVars(Set<Variable> bound, Set<Variable> free) {
    if (!bound.contains(this)) {
      free.add(this);
    }
  }

  @Override
  public boolean isConstant() {
    return false;
  }

  @Override
  public boolean isomorphic(Term b, Map<Variable, Variable> map) {

    // Variable?
    if (!(b instanceof Variable)) {
      return false;
    }
    var b1 = (Variable) b;

    // Equal?
    if (this == b1) {
      return true;
    }

    // Type match?
    if (!type.equals(b1.type)) {
      return false;
    }

    // Existing mapping
    var a2 = map.get(this);
    var b2 = map.get(b1);

    // Compatible mapping?
    if ((this == b2) && (b1 == a2)) {
      return true;
    }

    // New mapping?
    if ((a2 == null) && (b2 == null)) {
      map.put(this, b1);
      map.put(b1, this);
      return true;
    }

    // Incompatible mapping
    return false;
  }

  @Override
  public boolean match(Term b, Map<Variable, Term> map) {

    // Equal?
    if (this == b) {
      return true;
    }

    // Type match?
    if (!type().equals(b.type())) {
      return false;
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

  public String name() {
    return name;
  }

  @Override
  public Term replaceVars(Map<Variable, Term> map) {
    var a = map.get(this);
    if (a != null) {
      return a.replaceVars(map);
    }
    return this;
  }

  public void setName(String name) {
    this.name = name;
  }

  @Override
  public Tag tag() {
    return Tag.VAR;
  }

  @Override
  public String toString() {
    return (name() == null) ? '%' + Integer.toHexString(hashCode()) : name();
  }

  @Override
  public Type type() {
    return type;
  }

  @Override
  public void typeAssign(Map<TypeVariable, Type> map) {
    type = type.replaceVars(map);
    if (type instanceof TypeVariable) {
      type = Type.INDIVIDUAL;
    }
  }

  @Override
  public boolean unify(Term b, Map<Variable, Term> map) {

    // Equal?
    if (this == b) {
      return true;
    }

    // Type match?
    if (!type().equals(b.type())) {
      return false;
    }

    // Existing mapping
    var a2 = map.get(this);
    if (a2 != null) {
      return a2.unify(b, map);
    }

    // Variable?
    if (b instanceof Variable) {
      var b2 = map.get(b);
      if (b2 != null) {
        return unify(b2, map);
      }
    }

    // Occurs check
    if (b.contains(this, map)) {
      return false;
    }

    // New mapping
    map.put(this, b);
    return true;
  }
}
