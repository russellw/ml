package prover;

import java.util.Objects;

public abstract class Term1 extends Term {
    public final Term arg;

    public Term1(Term arg) {
        this.arg = arg;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if ((o == null) || (getClass() != o.getClass())) {
            return false;
        }
        Term1 b = (Term1) o;
        return Objects.equals(arg, b.arg);
    }

    @Override
    public Term get(int i) {
        if (i == 0) {
            return arg;
        }
        throw new IllegalArgumentException(toString() + '[' + i + ']');
    }

    @Override
    public int hashCode() {
        return Objects.hash(arg);
    }

    @Override
    public int size() {
        return 1;
    }
}
