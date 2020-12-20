package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.HashMap;

public abstract class Pattern {
  private final Object input;

  public Pattern(Object... input) {
    assert input.length > 1;
    this.input = Array.of(input);
  }

  public Object transform(Object a) {
    var map = Code.match(input, a, HashMap.empty());
    if (map == null) return a;
    return Code.replace(output(), map);
  }

  abstract Object output();
}
