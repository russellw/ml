package prover;

import java.util.*;

public abstract class Term implements Iterable<Term> {
  public static final Term FALSE =
      new Term() {

        @Override
        public boolean isBoolean() {
          return true;
        }

        @Override
        public Tag tag() {
          return Tag.FALSE;
        }
      };
  public static final Term TRUE =
      new Term() {

        @Override
        public boolean isBoolean() {
          return true;
        }

        @Override
        public Tag tag() {
          return Tag.TRUE;
        }
      };

  public boolean contains(Variable x, Map<Variable, Term> map) {
    for (var a : this) {
      if (a.contains(x, map)) {
        return true;
      }
    }
    return false;
  }

  public Term eval(Map<Variable, Term> map) {
    return this;
  }

  public Term get(int i) {
    throw new UnsupportedOperationException(toString());
  }

  @Override
  public final Iterator<Term> iterator() {
    return new Iterator<>() {
      private int i;

      @Override
      public boolean hasNext() {
        return i < size();
      }

      @Override
      public Term next() {
        return get(i++);
      }
    };
  }

  public boolean match(Term b, Map<Variable, Term> map) {
    if (equals(b)) {
      return true;
    }

    // Atoms unequal
    int size = size();
    if (size == 0) {
      return false;
    }

    // Structure
    if (tag() != b.tag()) {
      return false;
    }
    if (size != b.size()) {
      return false;
    }

    // Elements
    for (var i = 0; i < size; i++) {
      if (!get(i).match(b.get(i), map)) {
        return false;
      }
    }
    return true;
  }

  public boolean isBoolean() {
    return false;
  }

  public static Term of(boolean value) {
    return value ? TRUE : FALSE;
  }

  public static Term[] remove(Term[] terms, int i) {
    var r = new Term[terms.length - 1];
    System.arraycopy(terms, 0, r, 0, i);
    System.arraycopy(terms, i + 1, r, i, r.length - i);
    return r;
  }

  public Term rename(Map<Variable, Variable> map) {
    return transform(a -> a.rename(map));
  }

  public Term replace(Map<Variable, Term> map) {
    return transform(a -> a.replace(map));
  }

  public int size() {
    return 0;
  }

  public final Term splice(List<Integer> position, Term b) {
    return splice(position, 0, b);
  }

  public Term splice(List<Integer> position, int i, Term b) {
    if (i == position.size()) {
      return b;
    }
    throw new IllegalStateException(toString());
  }

  public abstract Tag tag();

  @Override
  public String toString() {
    var sb = new StringBuilder();
    sb.append(tag());
    if (size() > 0) {
      sb.append('(');
      for (int i = 0; i < size(); i++) {
        if (i > 0) {
          sb.append(",");
        }
        sb.append(get(i));
      }
      sb.append(')');
    }
    return sb.toString();
  }

  public Term transform(java.util.function.Function<Term, Term> f) {
    assert size() == 0;
    return this;
  }

  public static Term[] transform(Term[] q, java.util.function.Function<Term, Term> f) {
    var r = new Term[q.length];
    for (var i = 0; i < r.length; i++) {
      r[i] = f.apply(q[i]);
    }
    return r;
  }

  public boolean unify(Term b, Map<Variable, Term> map) {
    if (equals(b)) {
      return true;
    }
    if (b instanceof Variable) {
      return b.unify(this, map);
    }

    // Atoms unequal
    int size = size();
    if (size == 0) {
      return false;
    }

    // Structure
    if (tag() != b.tag()) {
      return false;
    }
    if (size != b.size()) {
      return false;
    }

    // Elements
    for (var i = 0; i < size; i++) {
      if (!get(i).unify(b.get(i), map)) {
        return false;
      }
    }
    return true;
  }
}
