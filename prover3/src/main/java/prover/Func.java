package prover;

import java.util.List;

public final class Func {
  public Object type;
  public String name;

  public Func(Object type, String name) {
    this.type = type;
    this.name = name;
  }

  public List<Object> call(Object a) {
    return List.of(this, a);
  }

  public List<Object> call(Object a, Object b) {
    return List.of(this, a, b);
  }

  @Override
  public String toString() {
    if (name == null) return String.format("_%x", hashCode());
    return name;
  }
}
