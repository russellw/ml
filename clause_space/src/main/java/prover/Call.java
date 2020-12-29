package prover;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public final class Call extends Term {
  private final Term[] data;

  public Call(Term[] data) {
    assert data.length > 1;
    this.data = data;
  }

  @Override
  public Term get(int i) {
    return data[i];
  }

  @Override
  public int size() {
    return data.length;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Call)) return false;
    Call terms = (Call) o;
    return Arrays.equals(data, terms.data);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(data);
  }

  @Override
  public boolean isBoolean() {
    return get(0).isBoolean();
  }

  @Override
  public Term splice(List<Integer> position, int i, Term b) {
    if (i == position.size()) return b;
    var r = new Term[size()];
    for (var j = 0; j < r.length; j++) {
      var x = get(j);
      if (j == position.get(i)) x = x.splice(position, i + 1, b);
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
      if (i > 1) sb.append(',');
      sb.append(get(i));
    }
    sb.append(')');
    return sb.toString();
  }

  @Override
  public Term transform(Function<Term, Term> f) {
    var r = new Term[size()];
    for (var i = 0; i < size(); i++) r[i] = f.apply(get(i));
    return new Call(r);
  }
}
