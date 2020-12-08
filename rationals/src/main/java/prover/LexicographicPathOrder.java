package prover;

import java.util.*;

public final class LexicographicPathOrder {
    private Map<Term, Integer> weights = new HashMap<>();

    public LexicographicPathOrder(Iterable<Term> ops) {
        var i = Tag.values().length;
        for (var a : ops) {
            weights.put(a, i++);
        }
    }

    public boolean greater(Term a, Term b) {
        assert !(a instanceof Function) || ((Function) a).arity() == 0;
        assert !(b instanceof Function) || ((Function) b).arity() == 0;

        // Fast equality test
        if (a == b) {
            return false;
        }

        // Variables are unordered unless contained in other term
        if (a instanceof Variable) {
            return false;
        }
        if (b instanceof Variable) {
            return a.contains((Variable) b);
        }

        // Sufficient condition: exists ai >= b
        for (var i = 1; i < a.size(); i++) {
            if (greaterEq(a.get(i), b)) {
                return true;
            }
        }

        // Necessary condition: a > all bi
        for (var i = 1; i < b.size(); i++) {
            if (!greater(a, b.get(i))) {
                return false;
            }
        }

        // Different function symbols
        var wa = weight(a);
        var wb = weight(b);
        if (wa != wb) {
            return wa > wb;
        }

        // Same weights means similar terms
        assert a.tag() == b.tag();
        assert a.size() == b.size();
        assert(a.size() == 0) || a.get(0).equals(b.get(0));

        // Constants
        switch (a.tag()) {
        case CONST_INTEGER:
            return a.integerValue().compareTo(b.integerValue()) > 0;
        case CONST_RATIONAL:
            return a.rationalValue().compareTo(b.rationalValue()) > 0;
        case CONST_REAL:
            return a.realValue() > b.realValue();
        case DISTINCT_OBJECT:
            return a.toString().compareTo(b.toString()) > 0;
        }

        // Lexicographic extension
        for (var i = 1; i < a.size(); i++) {
            var ai = a.get(i);
            var bi = b.get(i);
            if (greater(ai, bi)) {
                return true;
            }
            if (!ai.equals(bi)) {
                return false;
            }
        }
        if (!a.equals(b)) {
            throw new IllegalStateException(a + " != " + b);
        }
        return false;
    }

    private boolean greaterEq(Term a, Term b) {
        return greater(a, b) || a.equals(b);
    }

    private int weight(Term a) {
        var tag = a.tag();
        switch (tag) {
        case ATOM:
        case FUNC:
            return weights.get(a);
        case CONST_INTEGER:
        case CONST_RATIONAL:
        case CONST_REAL:
        case DISTINCT_OBJECT:
            return tag.ordinal();
        case CONST_TRUE:
            return -1;
        case LIST:
            return weights.get(a.get(0));
        default:
            throw new IllegalArgumentException(a.toString());
        }
    }
}
