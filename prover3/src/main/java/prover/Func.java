package prover;

import io.vavr.collection.Array;
import io.vavr.collection.Seq;

public final class Func {
  public Object type;
  public String name;

  public Func(Object type, String name) {
    this.type = type;
    this.name = name;
  }

  public Seq<Object> call(Object... args) {
    var r = new Object[args.length + 1];
    r[0] = this;
    System.arraycopy(args, 0, r, 1, args.length);
    return Array.of(r);
  }

  @Override
  public String toString() {
    return name;
  }
}
