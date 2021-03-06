package prover;

import java.util.List;

public final class Func extends Term {
  private final String name;
  public boolean isBoolean;

  public Func(String name) {
    this.name = name;
  }

  @Override
  public boolean isBoolean() {
    return isBoolean;
  }

  public Term call(List<? extends Term> args) {
    var r = new Term[args.size() + 1];
    r[0] = this;
    for (int i = 0; i < args.size(); i++) r[i + 1] = args.get(i);
    return new Call(r);
  }

  public Term call(Term... args) {
    var r = new Term[args.length + 1];
    r[0] = this;
    System.arraycopy(args, 0, r, 1, args.length);
    return new Call(r);
  }

  @Override
  public String toString() {
    return name;
  }
}
