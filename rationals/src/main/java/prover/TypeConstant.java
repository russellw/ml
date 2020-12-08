package prover;

public final class TypeConstant extends Type {
    public final String name;

    public TypeConstant(String name) {
        this.name = name;
    }

    @Override
    public Kind kind() {
        return Kind.CONSTANT;
    }
}
