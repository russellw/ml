package prover;

import java.util.*;

public abstract class Type implements Iterable<Type> {
  public static final Type BOOLEAN =
      new Type() {
        @Override
        public Kind kind() {
          return Kind.BOOLEAN;
        }

        @Override
        public String toString() {
          return "$o";
        }
      };
  public static final Type INDIVIDUAL =
      new Type() {
        @Override
        public Kind kind() {
          return Kind.INDIVIDUAL;
        }

        @Override
        public String toString() {
          return "$i";
        }
      };
  public static final Type INTEGER =
      new Type() {
        @Override
        public Kind kind() {
          return Kind.INTEGER;
        }

        @Override
        public String toString() {
          return "$int";
        }
      };
  public static final Type RATIONAL =
      new Type() {
        @Override
        public Kind kind() {
          return Kind.RATIONAL;
        }

        @Override
        public String toString() {
          return "$rat";
        }
      };
  public static final Type REAL =
      new Type() {
        @Override
        public Kind kind() {
          return Kind.REAL;
        }

        @Override
        public String toString() {
          return "$real";
        }
      };

  public boolean contains(TypeVariable x, Map<TypeVariable, Type> map) {
    return false;
  }

  public Type get(int i) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterator<Type> iterator() {
    return new Iterator<>() {
      @Override
      public boolean hasNext() {
        return false;
      }

      @Override
      public Type next() {
        return null;
      }
    };
  }

  public abstract Kind kind();

  public static Type of(Collection<? extends Type> q) {
    return of(q.toArray(new Type[0]));
  }

  public static Type of(Type... types) {
    return new CompoundList(types);
  }

  public Type replaceVars(Map<TypeVariable, Type> map) {
    return this;
  }

  public int size() {
    return 0;
  }

  public boolean unify(Type b, Map<TypeVariable, Type> map) {
    if (equals(b)) {
      return true;
    }
    if (b instanceof TypeVariable) {
      return b.unify(this, map);
    }
    if (size() == 0) {
      return false;
    }
    if (size() != b.size()) {
      return false;
    }
    for (var i = 0; i < size(); i++) {
      if (!get(i).unify(b.get(i), map)) {
        return false;
      }
    }
    return true;
  }

  private static class Compound extends Type {
    @Override
    public final boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Compound)) {
        return false;
      }
      var b = (Compound) o;
      if (size() != b.size()) {
        return false;
      }
      for (var i = 0; i < size(); i++) {
        if (!get(i).equals(b.get(i))) {
          return false;
        }
      }
      return true;
    }

    @Override
    public final int hashCode() {
      int n = 1;
      for (var a : this) {
        n = 31 * n + a.hashCode();
      }
      return n;
    }

    @Override
    public final Iterator<Type> iterator() {
      return new Iterator<>() {
        private int i;

        @Override
        public boolean hasNext() {
          return i < size();
        }

        @Override
        public Type next() {
          return get(i++);
        }
      };
    }

    @Override
    public Kind kind() {
      return Kind.FUNCTION;
    }

    @Override
    public final String toString() {
      var sb = new StringBuilder();
      sb.append('(');
      for (var i = 0; i < size(); i++) {
        if (i > 0) {
          sb.append(' ');
        }
        sb.append(get(i));
      }
      sb.append(')');
      return sb.toString();
    }
  }

  private static final class CompoundList extends Compound {
    private final Type[] data;

    CompoundList(Type... data) {
      this.data = data;
    }

    @Override
    public Type get(int i) {
      return data[i];
    }

    @Override
    public int size() {
      return data.length;
    }
  }

  private static final class CompoundSlice extends Compound {
    private final CompoundList list;
    private final int offset;

    CompoundSlice(CompoundList list, int offset) {
      this.list = list;
      this.offset = offset;
    }

    @Override
    public Type get(int i) {
      return list.get(offset + i);
    }

    @Override
    public int size() {
      return list.size() - offset;
    }
  }
}
