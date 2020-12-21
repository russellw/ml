package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.Map;

public abstract class Pattern {
  private final Object input;

  public Pattern(Object... input) {
    assert input.length > 1;
    this.input = Array.of(input);
  }

  public Object transform(Object a, Map<Variable, Object> map) {
    map = Code.match(input, a, map);
    if (map == null) return null;
    return Code.replace(output(map), map);
  }

  abstract Object output(Map<Variable, Object> map);
}
