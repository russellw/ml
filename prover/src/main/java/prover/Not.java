package prover;

public final class Not extends Term1 {
    public Not(Term arg0) {
        super(arg0);
    }

    @Override
    public Tag tag() {
        return Tag.NOT;
    }

    @Override
    public Type type() {
        return Type.BOOLEAN;
    }
}
