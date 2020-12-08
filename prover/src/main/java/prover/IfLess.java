package prover;

import java.util.Map;

public final class IfLess extends Term {
    private final Term left;
    private final double right;
    private final Term ifTrue, ifFalse;

    public IfLess(Term left, double right, Term ifTrue, Term ifFalse) {
        assert left.type() == Type.NUMBER;
        assert ifTrue.type() == ifFalse.type();
        this.left = left;
        this.right = right;
        this.ifTrue = ifTrue;
        this.ifFalse = ifFalse;
    }

    @Override
    public Term eval(Map<Variable, Term> map) {
        var test = left.eval(map).number() < right;
        return (test
                ? ifTrue
                : ifFalse).eval(map);
    }

    @Override
    public Term get(int i) {
        switch (i) {
        case 0:
            return left;
        case 1:
            return new Number(right);
        case 2:
            return ifTrue;
        case 3:
            return ifFalse;
        }
        throw new IllegalArgumentException(toString() + '[' + i + ']');
    }

    @Override
    public int size() {
        return 4;
    }

    @Override
    public Tag tag() {
        return Tag.IF_LESS;
    }

    @Override
    public Type type() {
        return ifTrue.type();
    }
}
