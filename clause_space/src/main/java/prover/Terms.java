package prover;

import java.util.Arrays;

public abstract class Terms extends Term {
  private final Term[] data;

  Terms(Term[] data) {
    this.data = data;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if ((o == null) || (getClass() != o.getClass())) {
      return false;
    }
    Terms b = (Terms) o;
    return Arrays.equals(data, b.data);
  }

  @Override
  public Term get(int i) {
    return data[i];
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(data);
  }

  @Override
  public int size() {
    return data.length;
  }
}
