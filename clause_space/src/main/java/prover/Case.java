package prover;

import java.util.List;
import java.util.Map;

public final class Case extends Terms {
  public Case(List<Term> data) {
    this(data.toArray(new Term[0]));
  }

  public Case(Term[] data) {
    super(data);
    assert get(0).type().categories != null;
    var type = type();
    for (int i = 2; i < size(); i++) {
      assert get(i).type() == type;
    }
  }

  @Override
  public Term eval(Map<Variable, Term> map) {
    var category = (Category) get(0).eval(map);
    return get(category.index + 1).eval(map);
  }

  @Override
  public Tag tag() {
    return Tag.CASE;
  }

  @Override
  public Type type() {
    return get(1).type();
  }
}
