package prover;

import java.util.*;
import java.util.function.Function;

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

  public boolean isBoolean() {
    return false;
  }

  public static Term[] remove(Term[] terms, int i) {
    var r = new Term[terms.length - 1];
    System.arraycopy(terms, 0, r, 0, i);
    System.arraycopy(terms, i + 1, r, i, r.length - i);
    return r;
  }

  public Term rename(Map<Var, Var> map) {
    return transform(a -> a.rename(map));
  }

  public Term replace(Map<Var, Term> map) {
    return transform(a -> a.replace(map));
  }

  public int size() {
    return 0;
  }

  public final Term splice(List<Integer> position, Term b) {
    return splice(position, 0, b);
  }

  public Term splice(List<Integer> position, int i, Term b) {
    if (i == position.size()) return b;
    throw new IllegalStateException(toString());
  }

  public abstract Tag tag();

  public Term transform(Function<Term, Term> f) {
    assert size() == 0;
    return this;
  }

  public static Term[] transform(Term[] q, Function<Term, Term> f) {
    var r = new Term[q.length];
    for (var i = 0; i < r.length; i++) r[i] = f.apply(q[i]);
    return r;
  }
}
