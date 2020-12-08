package prover;

public final class TypeException extends RuntimeException {
    private static final long serialVersionUID = 0;
    public final Object term;
    public final Object actual;
    public final Object expected;

    public TypeException(Object term, Object actual) {
        super(term + ": type mismatch: " + actual);
        this.term = term;
        this.actual = actual;
        expected = null;
    }

    public TypeException(Object term, Object actual, Object expected) {
        super(term + ": type mismatch: " + actual + " != " + expected);
        this.term = term;
        this.actual = actual;
        this.expected = expected;
    }
}
