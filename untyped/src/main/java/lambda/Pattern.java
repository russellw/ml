package lambda;

import io.vavr.collection.Array;
import io.vavr.collection.Map;

public abstract class Pattern {
  private final Object input;

  public Pattern(Object... input) {
    assert input.length > 1;
    this.input = Array.of(input);
  }

  public Object transform(Map<Variable, Object> map, Object a) {
    map = Code.match(map, input, a);
    if (map == null) return null;
    return Code.replace(map, output(map));
  }

  abstract Object output(Map<Variable, Object> map);
}
