package prover;

public final class Category extends Term {
    private final Type type;
    private final String name;
    public final int index;

    public Category(Type type, String name, int index) {
        this.type = type;
        this.name = name;
        this.index = index;
    }

    @Override
    public Tag tag() {
        return Tag.CATEGORY;
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public Type type() {
        return type;
    }
}
