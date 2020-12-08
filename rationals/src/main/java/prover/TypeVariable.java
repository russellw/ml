package prover;

import java.util.Map;

public final class TypeVariable extends Type {
    @Override
    public boolean contains(TypeVariable x, Map<TypeVariable, Type> map) {
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
    public Kind kind() {
        return Kind.VARIABLE;
    }

    @Override
    public Type replaceVars(Map<TypeVariable, Type> map) {
        var a = map.get(this);
        if (a != null) {
            return a.replaceVars(map);
        }
        return this;
    }

    @Override
    public boolean unify(Type b, Map<TypeVariable, Type> map) {

        // Equal?
        if (this == b) {
            return true;
        }

        // Existing mapping
        var a2 = map.get(this);
        if (a2 != null) {
            return a2.unify(b, map);
        }

        // Variable?
        if (b instanceof TypeVariable) {
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
