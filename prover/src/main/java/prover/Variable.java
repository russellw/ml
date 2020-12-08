package prover;

import java.util.Map;

public final class Variable extends Term {
    private final Type type;
    private final String name;

    public Variable(Type type, String name) {
        this.type = type;
        this.name = name;
    }

    @Override
    public boolean contains(Variable x, Map<Variable, Term> map) {
        if (this == x) {
            return true;
        }
        var a = map.get(this);
        if (a != null) {
            return a.contains(x, map);
        }
        return false;
    }

    @Override
    public Term eval(Map<Variable, Term> map) {
        return map.get(this);
    }

    @Override
    public boolean match(Term b, Map<Variable, Term> map) {

        // Equal?
        if (this == b) {
            return true;
        }

        // Type match?
        if (type() != b.type()) {
            return false;
        }

        // Existing mapping
        var a2 = map.get(this);
        if (a2 != null) {
            return a2.equals(b);
        }

        // New mapping
        map.put(this, b);
        return true;
    }

    @Override
    public Term rename(Map<Variable, Variable> map) {
        var a = map.get(this);
        if (a == null) {
            a = new Variable(type, name);
            map.put(this, a);
        }
        return a;
    }

    @Override
    public Term replace(Map<Variable, Term> map) {
        var a = map.get(this);
        if (a != null) {
            return a.replace(map);
        }
        return this;
    }

    @Override
    public Tag tag() {
        return Tag.VARIABLE;
    }

    @Override
    public String toString() {
        if (name == null) {
            return Integer.toHexString(hashCode());
        }
        return name;
    }

    @Override
    public Type type() {
        return type;
    }

    @Override
    public boolean unify(Term b, Map<Variable, Term> map) {

        // Equal?
        if (this == b) {
            return true;
        }

        // Type match?
        if (type() != b.type()) {
            return false;
        }

        // Existing mapping
        var a2 = map.get(this);
        if (a2 != null) {
            return a2.unify(b, map);
        }

        // Variable?
        if (b instanceof Variable) {
            var b2 = map.get(b);
            if (b2 != null) {
                return unify(b2, map);
            }
        }

        // Occurs check
        if (b.contains(this, map)) {
            return false;
        }

        // New mapping
        map.put(this, b);
        return true;
    }
}
