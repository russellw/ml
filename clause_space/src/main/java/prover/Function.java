package prover;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Function extends Term {
  public static final Map<String, Variable> COMMON_NAMES = new HashMap<>();
  private Type type;
  private final String name;
  public final Variable variable;

  public Function(String name) {
    this.name = name;
    variable = COMMON_NAMES.get(name);
  }

  public Term call(List<? extends Term> args) {
    var r = new Term[args.size() + 1];
    r[0] = this;
    for (int i = 0; i < args.size(); i++) {
      r[i + 1] = args.get(i);
    }
    return new Call(r);
  }

  public Term call(Term... args) {
    var r = new Term[args.length + 1];
    r[0] = this;
    System.arraycopy(args, 0, r, 1, args.length);
    return new Call(r);
  }

  @Override
  public Tag tag() {
    return Tag.FUNCTION;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public Type type() {
    if (type == null) {
      return Type.INDIVIDUAL;
    }
    return type;
  }

  @Override
  public void type(Type expected) {
    if (type == null) {
      type = expected;
      return;
    }
    if (type() != expected) {
      throw new IllegalStateException(this + ": " + type() + " != " + expected);
    }
  }
}
