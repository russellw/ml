package prover;

import java.util.List;
import java.util.function.Function;

public final class Call extends Terms {
  public Call(Term[] data) {
    super(data);
    assert data.length > 0;
  }

  @Override
  public boolean isBoolean() {
    return get(0).isBoolean();
  }

  @Override
  public Term splice(List<Integer> position, int i, Term b) {
    if (i == position.size()) {
      return b;
    }
    var r = new Term[size()];
    for (var j = 0; j < r.length; j++) {
      var x = get(j);
      if (j == position.get(i)) {
        x = x.splice(position, i + 1, b);
      }
      r[j] = x;
    }
    return new Call(r);
  }

  @Override
  public Tag tag() {
    return Tag.CALL;
  }

  @Override
  public String toString() {
    var sb = new StringBuilder();
    sb.append(get(0));
    sb.append('(');
    for (int i = 1; i < size(); i++) {
      if (i > 1) {
        sb.append(",");
      }
      sb.append(get(i));
    }
    sb.append(')');
    return sb.toString();
  }

  @Override
  public Term transform(Function<Term, Term> f) {
    var r = new Term[size()];
    for (var i = 0; i < size(); i++) {
      r[i] = f.apply(get(i));
    }
    return new Call(r);
  }
}
