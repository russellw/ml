package prover;

public final class Func {
  public Object type;
  public String name;

  public Func(Object type, String name) {
    this.type = type;
    this.name = name;
  }

  @Override
  public String toString() {
    if (name == null) return String.format("_%x", hashCode());
    return name;
  }
}
