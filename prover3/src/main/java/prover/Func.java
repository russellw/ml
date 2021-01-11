package prover;

import java.util.List;

public final class Func {
  public Object type;
  public String name;

  public Func(Object type, String name) {
    this.type = type;
    this.name = name;
  }

  public List<Object> call(Object... args) {
    var r = new Object[args.length + 1];
    r[0] = this;
    System.arraycopy(args, 0, r, 1, args.length);
    return List.of(r);
  }

  @Override
  public String toString() {
    if (name == null) return String.format("_%x", hashCode());
    return name;
  }
}
