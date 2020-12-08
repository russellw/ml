package prover;

import java.math.BigDecimal;
import java.math.BigInteger;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Predicate;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import static prover.Type.*;

public abstract class Term implements Iterable<Term> {

    // Constants
    public static final Term FALSE = new Term() {
        @Override
        public Term eval(Map<Variable, Term> map) {
            return this;
        }
        @Override
        public Tag tag() {
            return Tag.CONST_FALSE;
        }
        @Override
        public Type type() {
            return Type.BOOLEAN;
        }
    };
    public static final Term TRUE = new Term() {
        @Override
        public Term eval(Map<Variable, Term> map) {
            return this;
        }
        @Override
        public Tag tag() {
            return Tag.CONST_TRUE;
        }
        @Override
        public Type type() {
            return Type.BOOLEAN;
        }
    };

    // Operators
    public static final Term ADD = new Term() {
        @Override
        public Op called() {
            return Op.ADD;
        }
    };
    public static final Term ALL = new Term() {
        @Override
        public Op called() {
            return Op.ALL;
        }
    };
    public static final Term AND = new Term() {
        @Override
        public Op called() {
            return Op.AND;
        }
    };
    public static final Term CEIL = new Term() {
        @Override
        public Op called() {
            return Op.CEIL;
        }
    };
    public static final Term DIVIDE = new Term() {
        @Override
        public Op called() {
            return Op.DIVIDE;
        }
    };
    public static final Term DIVIDE_EUCLIDEAN = new Term() {
        @Override
        public Op called() {
            return Op.DIVIDE_EUCLIDEAN;
        }
    };
    public static final Term DIVIDE_FLOOR = new Term() {
        @Override
        public Op called() {
            return Op.DIVIDE_FLOOR;
        }
    };
    public static final Term DIVIDE_TRUNCATE = new Term() {
        @Override
        public Op called() {
            return Op.DIVIDE_TRUNCATE;
        }
    };
    public static final Term EQ = new Term() {
        @Override
        public Op called() {
            return Op.EQ;
        }
    };
    public static final Term EQV = new Term() {
        @Override
        public Op called() {
            return Op.EQV;
        }
    };
    public static final Term EXISTS = new Term() {
        @Override
        public Op called() {
            return Op.EXISTS;
        }
    };
    public static final Term FLOOR = new Term() {
        @Override
        public Op called() {
            return Op.FLOOR;
        }
    };
    public static final Term IF = new Term() {
        @Override
        public Op called() {
            return Op.IF;
        }
    };
    public static final Term IS_INTEGER = new Term() {
        @Override
        public Op called() {
            return Op.IS_INTEGER;
        }
    };
    public static final Term IS_RATIONAL = new Term() {
        @Override
        public Op called() {
            return Op.IS_RATIONAL;
        }
    };
    public static final Term LESS = new Term() {
        @Override
        public Op called() {
            return Op.LESS;
        }
    };
    public static final Term LESS_EQ = new Term() {
        @Override
        public Op called() {
            return Op.LESS_EQ;
        }
    };
    public static final Term MULTIPLY = new Term() {
        @Override
        public Op called() {
            return Op.MULTIPLY;
        }
    };
    public static final Term NEGATE = new Term() {
        @Override
        public Op called() {
            return Op.NEGATE;
        }
    };
    public static final Term NOT = new Term() {
        @Override
        public Op called() {
            return Op.NOT;
        }
    };
    public static final Term OR = new Term() {
        @Override
        public Op called() {
            return Op.OR;
        }
    };
    public static final Term REMAINDER_EUCLIDEAN = new Term() {
        @Override
        public Op called() {
            return Op.REMAINDER_EUCLIDEAN;
        }
    };
    public static final Term REMAINDER_FLOOR = new Term() {
        @Override
        public Op called() {
            return Op.REMAINDER_FLOOR;
        }
    };
    public static final Term REMAINDER_TRUNCATE = new Term() {
        @Override
        public Op called() {
            return Op.REMAINDER_TRUNCATE;
        }
    };
    public static final Term ROUND = new Term() {
        @Override
        public Op called() {
            return Op.ROUND;
        }
    };
    public static final Term SUBTRACT = new Term() {
        @Override
        public Op called() {
            return Op.SUBTRACT;
        }
    };
    public static final Term TO_INTEGER = new Term() {
        @Override
        public Op called() {
            return Op.TO_INTEGER;
        }
    };
    public static final Term TO_RATIONAL = new Term() {
        @Override
        public Op called() {
            return Op.TO_RATIONAL;
        }
    };
    public static final Term TO_REAL = new Term() {
        @Override
        public Op called() {
            return Op.TO_REAL;
        }
    };
    public static final Term TRUNCATE = new Term() {
        @Override
        public Op called() {
            return Op.TRUNCATE;
        }
    };

    public final Term add(Term b) {
        return of(ADD, this, b);
    }

    public final boolean allMatch(Predicate<Term> f) {
        for (var a : this) {
            if (!f.test(a)) {
                return false;
            }
        }
        return true;
    }

    public final Term and(Term b) {
        return of(AND, this, b);
    }

    public Op called() {
        return Op.CALL;
    }

    public Term cdr() {
        throw new UnsupportedOperationException(toString());
    }

    public final Term ceil() {
        return of(CEIL, this);
    }

    public boolean contains(Variable x) {
        for (var a : this) {
            if (a.contains(x)) {
                return true;
            }
        }
        return false;
    }

    public boolean contains(Variable x, Map<Variable, Term> map) {
        for (var a : this) {
            if (a.contains(x, map)) {
                return true;
            }
        }
        return false;
    }

    public final Term deepGet(int i) {
        if (i == 0) {
            return this;
        }
        var n = 1;
        for (var term : cdr()) {
            var j = term.deepSize();
            if (i < n + j) {
                return term.deepGet(i - n);
            }
            n += j;
        }
        throw new IllegalArgumentException(this + ": " + i);
    }

    public final int deepSize() {
        if (size() == 0) {
            return 1;
        }
        var n = 1;
        for (var term : cdr()) {
            n += term.deepSize();
        }
        return n;
    }

    public final Term deepSplice(int i, Term b) {
        if (i == 0) {
            return b;
        }
        var n = 1;
        var r = new ArrayList<Term>();
        r.add(get(0));
        for (var x : cdr()) {
            var j = x.deepSize();
            r.add(((n <= i) && (i < n + j))
                  ? x.deepSplice(i - n, b)
                  : x);
            n += j;
        }
        return of(r);
    }

    public final int depth() {
        if (size() == 0) {
            return 0;
        }
        var n = 0;
        for (var term : this) {
            n = Math.max(n, term.depth());
        }
        return n + 1;
    }

    public final Term divide(Term b) {
        return of(DIVIDE, this, b);
    }

    public final Term divideEuclidean(Term b) {
        return of(DIVIDE_EUCLIDEAN, this, b);
    }

    public final Term divideFloor(Term b) {
        return of(DIVIDE_FLOOR, this, b);
    }

    public final Term divideTruncate(Term b) {
        return of(DIVIDE_TRUNCATE, this, b);
    }

    public final Term eq(Term b) {
        return of(EQ, this, b);
    }

    public final Term eqv(Term b) {
        return of(EQV, this, b);
    }

    public Term eval(Map<Variable, Term> map) {
        switch (op()) {
        case ADD:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.add(b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.add(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a + b);
                             }
                         });
        case AND:
            for (var a : cdr()) {
                if (a.eval(map) == FALSE) {
                    return FALSE;
                }
            }
            return TRUE;
        case CEIL: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return a;
            case CONST_RATIONAL:
                return of(BigRational.of(a.rationalValue().ceil()));
            case CONST_REAL:
                return of(Math.ceil(a.realValue()));
            }
            break;
        }
        case DIVIDE:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(BigRational.of(a, b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.divide(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a / b);
                             }
                         });
        case DIVIDE_EUCLIDEAN:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(Util.divideEuclidean(a, b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.divideEuclidean(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case DIVIDE_FLOOR:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(Util.divideFloor(a, b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.divideFloor(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case DIVIDE_TRUNCATE:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.divide(b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.divideTruncate(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case EQV:
        case EQ: {
            var a = get(1).eval(map);
            var b = get(2).eval(map);
            return of(a.equals(b));
        }
        case FLOOR: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return a;
            case CONST_RATIONAL:
                return of(BigRational.of(a.rationalValue().floor()));
            case CONST_REAL:
                return of(Math.floor(a.realValue()));
            }
            break;
        }
        case IF: {
            var test = get(1).eval(map);
            return (test == TRUE)
                   ? get(2).eval(map)
                   : get(3).eval(map);
        }
        case NOT: {
            var a = get(1).eval(map);
            return of(a == FALSE);
        }
        case IS_INTEGER: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return TRUE;
            case CONST_RATIONAL:
                return of(a.rationalValue().den.equals(BigInteger.ONE));
            case CONST_REAL:
                return of(Util.isInteger(a.realValue()));
            }
            break;
        }
        case IS_RATIONAL:
            return TRUE;
        case LESS:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.compareTo(b) < 0);
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.compareTo(b) < 0);
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a < b);
                             }
                         });
        case LESS_EQ:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.compareTo(b) <= 0);
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.compareTo(b) <= 0);
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a <= b);
                             }
                         });
        case MULTIPLY:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.multiply(b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.multiply(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a * b);
                             }
                         });
        case NEGATE: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return of(a.integerValue().negate());
            case CONST_RATIONAL:
                return of(a.rationalValue().negate());
            case CONST_REAL:
                return of(-a.realValue());
            }
            break;
        }
        case OR:
            for (var a : cdr()) {
                if (a.eval(map) == TRUE) {
                    return TRUE;
                }
            }
            return FALSE;
        case REMAINDER_EUCLIDEAN:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(Util.remainderEuclidean(a, b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.remainderEuclidean(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case REMAINDER_FLOOR:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(Util.remainderFloor(a, b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.remainderFloor(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case REMAINDER_TRUNCATE:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.remainder(b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.remainderTruncate(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 throw new IllegalArgumentException(a + "," + b);
                             }
                         });
        case ROUND: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return a;
            case CONST_RATIONAL:
                return of(BigRational.of(a.rationalValue().round()));
            case CONST_REAL:
                return of(Math.round(a.realValue()));
            }
            break;
        }
        case SUBTRACT:
            return eval2(map,
                         new Op2() {
                             @Override
                             Term apply(BigInteger a, BigInteger b) {
                                 return of(a.subtract(b));
                             }
                             @Override
                             Term apply(BigRational a, BigRational b) {
                                 return of(a.subtract(b));
                             }
                             @Override
                             Term apply(double a, double b) {
                                 return of(a - b);
                             }
                         });
        case TRUNCATE: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return a;
            case CONST_RATIONAL:
                return of(BigRational.of(a.rationalValue().truncate()));
            case CONST_REAL:
                return of(Util.truncate(a.realValue()));
            }
            break;
        }
        case TO_INTEGER: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return a;
            case CONST_RATIONAL:
                return of(a.rationalValue().floor());
            case CONST_REAL:
                return of(BigDecimal.valueOf(a.realValue()).toBigInteger());
            }
            break;
        }
        case TO_RATIONAL: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return of(BigRational.of(a.integerValue()));
            case CONST_RATIONAL:
                return a;
            case CONST_REAL:
                return of(BigRational.of(a.realValue()));
            }
            break;
        }
        case TO_REAL: {
            var a = get(1).eval(map);
            switch (a.tag()) {
            case CONST_INTEGER:
                return of(a.integerValue().doubleValue());
            case CONST_RATIONAL:
                return of(a.rationalValue().doubleValue());
            case CONST_REAL:
                return a;
            }
            break;
        }
        }
        throw new IllegalStateException(toString());
    }

    private Term eval2(Map<Variable, Term> map, Op2 op) {
        var a = get(1).eval(map);
        var b = get(2).eval(map);
        switch (a.tag()) {
        case CONST_INTEGER:
            switch (b.tag()) {
            case CONST_INTEGER:
                return op.apply(a.integerValue(), b.integerValue());
            case CONST_RATIONAL:
                return op.apply(BigRational.of(a.integerValue()), b.rationalValue());
            case CONST_REAL:
                return op.apply(a.integerValue().doubleValue(), b.realValue());
            }
            break;
        case CONST_RATIONAL:
            switch (b.tag()) {
            case CONST_INTEGER:
                return op.apply(a.rationalValue(), BigRational.of(b.integerValue()));
            case CONST_RATIONAL:
                return op.apply(a.rationalValue(), b.rationalValue());
            case CONST_REAL:
                return op.apply(a.rationalValue().doubleValue(), b.realValue());
            }
            break;
        case CONST_REAL:
            switch (b.tag()) {
            case CONST_INTEGER:
                return op.apply(a.realValue(), b.integerValue().doubleValue());
            case CONST_RATIONAL:
                return op.apply(a.realValue(), b.rationalValue().doubleValue());
            case CONST_REAL:
                return op.apply(a.realValue(), b.realValue());
            }
            break;
        }
        throw new IllegalStateException(toString());
    }

    public final Term floor() {
        return of(FLOOR, this);
    }

    public final Set<Variable> freeVars() {
        Set<Variable> free = new LinkedHashSet<>();
        getVars(new HashSet<>(), free);
        return free;
    }

    public Term freshVars(Map<Variable, Variable> map) {
        return map(a -> a.freshVars(map));
    }

    public final Set<Function> functions() {
        var r = new HashSet<Function>();
        getFuncs(r);
        return r;
    }

    public Term get(int i) {
        throw new UnsupportedOperationException(toString());
    }

    public void getFuncs(Set<Function> r) {
        for (var term : this) {
            term.getFuncs(r);
        }
    }

    public void getOps(Set<Term> r) {
        if (size() == 0) {
            return;
        }
        r.add(get(0));
        for (var term : cdr()) {
            term.getOps(r);
        }
    }

    public void getVars(Set<Variable> r) {
        for (var term : this) {
            term.getVars(r);
        }
    }

    public void getVars(Set<Variable> bound, Set<Variable> free) {
        if (isQuantifier()) {
            var bound1 = new HashSet<>(bound);
            for (var x : get(1)) {
                bound1.add((Variable) x);
            }
            get(2).getVars(bound1, free);
            return;
        }
        for (var term : this) {
            term.getVars(bound, free);
        }
    }

    public final Term implies(Term b) {
        return of(OR, not(), b);
    }

    public BigInteger integerValue() {
        throw new UnsupportedOperationException(toString());
    }

    public boolean isConstant() {
        return true;
    }

    public boolean isOne() {
        return false;
    }

    public final boolean isQuantifier() {
        switch (op()) {
        case ALL:
        case EXISTS:
            return true;
        }
        return false;
    }

    public boolean isZero() {
        return false;
    }

    public boolean isomorphic(Term b, Map<Variable, Variable> map) {
        if (equals(b)) {
            return true;
        }
        if (size() == 0) {
            return false;
        }
        if (!type().equals(b.type())) {
            return false;
        }
        if (size() != b.size()) {
            return false;
        }
        if (isQuantifier()) {
            if (get(0) != b.get(0)) {
                return false;
            }
            var a1 = get(1);
            var b1 = b.get(1);
            if (a1.size() != b1.size()) {
                return false;
            }
            for (var i = 0; i < a1.size(); i++) {
                if (!a1.get(i).isomorphic(b1.get(i), map)) {
                    return false;
                }
            }
            return get(2).isomorphic(b.get(2), map);
        }
        for (var i = 0; i < size(); i++) {
            if (!get(i).isomorphic(b.get(i), map)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Iterator<Term> iterator() {
        return new Iterator<>() {
            @Override
            public boolean hasNext() {
                return false;
            }
            @Override
            public Term next() {
                return null;
            }
        };
    }

    public final Term less(Term b) {
        return of(LESS, this, b);
    }

    public final Term lessEq(Term b) {
        return of(LESS_EQ, this, b);
    }

    public final Term map(java.util.function.Function<Term, Term> f) {
        if (size() == 0) {
            return this;
        }
        var r = new Term[size()];
        for (var i = 0; i < size(); i++) {
            r[i] = f.apply(get(i));
        }
        return of(r);
    }

    public static Term[] map(Term[] q, java.util.function.Function<Term, Term> f) {
        var r = new Term[q.length];
        for (var i = 0; i < r.length; i++) {
            r[i] = f.apply(q[i]);
        }
        return r;
    }

    public final Term map1(java.util.function.Function<Term, Term> f) {
        if (size() == 0) {
            return this;
        }
        var r = new Term[size()];
        r[0] = get(0);
        for (var i = 1; i < size(); i++) {
            r[i] = f.apply(get(i));
        }
        return of(r);
    }

    public boolean match(Term b, Map<Variable, Term> map) {
        if (equals(b)) {
            return true;
        }
        if (size() == 0) {
            return false;
        }
        if (!type().equals(b.type())) {
            return false;
        }
        if (size() != b.size()) {
            return false;
        }
        for (var i = 0; i < size(); i++) {
            if (!get(i).match(b.get(i), map)) {
                return false;
            }
        }
        return true;
    }

    public final Term multiply(Term b) {
        return of(MULTIPLY, this, b);
    }

    public final Term nand(Term b) {
        return and(b).not();
    }

    public final Term negate() {
        return of(NEGATE, this);
    }

    public final Term nor(Term b) {
        return or(b).not();
    }

    public final Term not() {
        return of(NOT, this);
    }

    public final Term notEq(Term b) {
        return eq(b).not();
    }

    public static Term of(BigInteger value) {
        return new ConstInteger(value);
    }

    public static Term of(BigRational value) {
        return new ConstRational(value);
    }

    public static Term of(boolean value) {
        return value
               ? TRUE
               : FALSE;
    }

    public static Term of(Collection<? extends Term> q) {
        return of(q.toArray(new Term[0]));
    }

    public static Term of(double value) {
        return new ConstReal(value);
    }

    public static Term of(long value) {
        return of(BigInteger.valueOf(value));
    }

    public static Term of(Term... terms) {
        if (terms.length == 2) {
            switch (terms[0].called()) {
            case AND:
            case OR:
                return terms[1];
            }
        }
        return new CompoundList(terms);
    }

    public static Term of(Term op, Collection<? extends Term> terms) {
        var r = new Term[1 + terms.size()];
        r[0] = op;
        var i = 1;
        for (var a : terms) {
            r[i++] = a;
        }
        return of(r);
    }

    public Op op() {
        return Op.ATOM;
    }

    public final Term or(Term b) {
        return of(OR, this, b);
    }

    private void print(int indent) {
        if (this.depth() <= 4) {
            System.out.print(this);
            return;
        }
        System.out.print('(');
        System.out.print(get(0));
        indent++;
        for (var i = 1; i < size(); i++) {
            var term = get(i);
            System.out.println();
            for (var j = 0; j < indent; j++) {
                System.out.print(' ');
            }
            term.print(indent);
        }
        System.out.print(')');
    }

    public final void println() {
        print(0);
        System.out.println();
    }

    public final Term quantify() {
        var free = freeVars();
        if (free.isEmpty()) {
            return this;
        }
        return of(ALL, of(free), this);
    }

    public BigRational rationalValue() {
        throw new UnsupportedOperationException(toString());
    }

    public double realValue() {
        throw new UnsupportedOperationException(toString());
    }

    public final Term remainderEuclidean(Term b) {
        return of(REMAINDER_EUCLIDEAN, this, b);
    }

    public final Term remainderFloor(Term b) {
        return of(REMAINDER_FLOOR, this, b);
    }

    public final Term remainderTruncate(Term b) {
        return of(REMAINDER_TRUNCATE, this, b);
    }

    public static Term[] remove(Term[] terms, int i) {
        var r = new Term[terms.length - 1];
        System.arraycopy(terms, 0, r, 0, i);
        System.arraycopy(terms, i + 1, r, i, r.length - i);
        return r;
    }

    public final Term replace(Map<Term, Term> map) {
        var a = map.get(this);
        if (a != null) {
            return a.replace(map);
        }
        return map(b -> b.replace(map));
    }

    public Term replaceVars(Map<Variable, Term> map) {
        return map(a -> a.replaceVars(map));
    }

    public final Term round() {
        return of(ROUND, this);
    }

    public final Term simplify() {
        if (size() == 0) {
            return this;
        }

        // Recur
        var a = map(Term::simplify);

        // In theorem context, don't actually convert to floating point
        if (a.get(0) == TO_REAL) {
            return a;
        }

        // Constant expressions can be fully evaluated
        if (a.allMatch(Term::isConstant)) {
            return a.eval(new HashMap<>());
        }

        // Non-constant expressions may have useful identities
        switch (a.op()) {
        case ADD: {
            var x = a.get(1);
            var y = a.get(2);
            if (x.isZero()) {
                return y;
            }
            if (y.isZero()) {
                return x;
            }
            break;
        }
        case AND: {
            var r = new ArrayList<Term>();
            r.add(a.get(0));
            for (var x : cdr()) {
                switch (x.tag()) {
                case CONST_FALSE:
                    return x;
                case CONST_TRUE:
                    continue;
                }
                r.add(x);
            }
            switch (r.size()) {
            case 1:
                return TRUE;
            case 2:
                return r.get(1);
            }
            return of(r);
        }
        case EQ:
        case LESS_EQ: {
            var x = a.get(1);
            var y = a.get(2);
            if (x.equals(y)) {
                return TRUE;
            }
            break;
        }
        case IF: {
            var test = a.get(1);
            switch (test.tag()) {
            case CONST_FALSE:
                return a.get(3);
            case CONST_TRUE:
                return a.get(2);
            }
            break;
        }
        case MULTIPLY: {
            var x = a.get(1);
            var y = a.get(2);
            if (x.isZero()) {
                return x;
            }
            if (y.isZero()) {
                return y;
            }
            if (x.isOne()) {
                return y;
            }
            if (y.isOne()) {
                return x;
            }
            break;
        }
        case OR: {
            var r = new ArrayList<Term>();
            r.add(a.get(0));
            for (var x : cdr()) {
                switch (x.tag()) {
                case CONST_FALSE:
                    continue;
                case CONST_TRUE:
                    return x;
                }
                r.add(x);
            }
            switch (r.size()) {
            case 1:
                return FALSE;
            case 2:
                return r.get(1);
            }
            return of(r);
        }
        case SUBTRACT: {
            var x = a.get(1);
            var y = a.get(2);
            if (x.isZero()) {
                return of(NEGATE, y);
            }
            if (y.isZero()) {
                return x;
            }
            break;
        }
        }
        return a;
    }

    public int size() {
        return 0;
    }

    public final Term splice(List<Integer> position, Term b) {
        return splice(position, 0, b);
    }

    private Term splice(List<Integer> position, int i, Term b) {
        if (i == position.size()) {
            return b;
        }
        assert size() > 0;
        var r = new Term[size()];
        for (var j = 0; j < r.length; j++) {
            var x = get(j);
            if (j == position.get(i)) {
                x = x.splice(position, i + 1, b);
            }
            r[j] = x;
        }
        return of(r);
    }

    public final Term subtract(Term b) {
        return of(SUBTRACT, this, b);
    }

    public Tag tag() {
        return Tag.ATOM;
    }

    public final Term toInteger() {
        return of(TO_INTEGER, this);
    }

    public final Term toRational() {
        return of(TO_RATIONAL, this);
    }

    public final Term toReal() {
        return of(TO_REAL, this);
    }

    @Override
    public String toString() {
        var tag = tag();
        if (tag != Tag.ATOM) {
            return tag.toString();
        }
        return called().toString();
    }

    public final Term truncate() {
        return of(TRUNCATE, this);
    }

    public Type type() {
        switch (op()) {
        case ADD:
        case CEIL:
        case DIVIDE:
        case DIVIDE_EUCLIDEAN:
        case DIVIDE_FLOOR:
        case DIVIDE_TRUNCATE:
        case FLOOR:
        case MULTIPLY:
        case NEGATE:
        case REMAINDER_EUCLIDEAN:
        case REMAINDER_FLOOR:
        case REMAINDER_TRUNCATE:
        case ROUND:
        case SUBTRACT:
        case TRUNCATE:
            return get(1).type();
        case ALL:
        case AND:
        case EQ:
        case EQV:
        case EXISTS:
        case IS_INTEGER:
        case IS_RATIONAL:
        case LESS:
        case LESS_EQ:
        case NOT:
        case OR:
            return Type.BOOLEAN;
        case CALL: {
            var funcType = get(0).type();
            if (funcType.size() == 0) {
                throw new IllegalStateException(funcType.toString() + ' ' + this);
            }
            return funcType.get(0);
        }
        case IF:
            return get(2).type();
        case TO_INTEGER:
            return INTEGER;
        case TO_RATIONAL:
            return RATIONAL;
        case TO_REAL:
            return REAL;
        }
        throw new IllegalStateException(tag().toString() + ' ' + toString());
    }

    public void typeAssign(Map<TypeVariable, Type> map) {
        for (var term : this) {
            term.typeAssign(map);
        }
    }

    public final void typeCheck(Type expected) {
        if (!type().equals(expected)) {
            throw new TypeException(this, type(), expected);
        }
        switch (op()) {
        case ADD:
        case CEIL:
        case DIVIDE:
        case DIVIDE_EUCLIDEAN:
        case DIVIDE_FLOOR:
        case DIVIDE_TRUNCATE:
        case FLOOR:
        case IS_INTEGER:
        case IS_RATIONAL:
        case LESS:
        case LESS_EQ:
        case MULTIPLY:
        case NEGATE:
        case REMAINDER_EUCLIDEAN:
        case REMAINDER_FLOOR:
        case REMAINDER_TRUNCATE:
        case ROUND:
        case SUBTRACT:
        case TO_INTEGER:
        case TO_RATIONAL:
        case TO_REAL:
        case TRUNCATE: {
            var argType = get(1).type();
            switch (argType.kind()) {
            case INTEGER:
            case RATIONAL:
            case REAL:
                break;
            default:
                throw new TypeException(this, argType);
            }
            for (var term : cdr()) {
                term.typeCheck(argType);
            }
            break;
        }
        case ALL:
        case EXISTS:
            get(2).typeCheck(Type.BOOLEAN);
            break;
        case AND:
        case EQV:
        case NOT:
        case OR:
            for (var term : cdr()) {
                term.typeCheck(Type.BOOLEAN);
            }
            break;
        case ATOM:
            break;
        case CALL: {
            var funcType = get(0).type();
            if (funcType.size() == 0) {
                throw new TypeException(this, funcType);
            }
            if (funcType.size() != size()) {
                throw new TypeException(this, funcType);
            }
            for (var i = 1; i < size(); i++) {
                get(i).typeCheck(funcType.get(i));
            }
            break;
        }
        case EQ: {
            var argType = get(1).type();
            get(1).typeCheck(argType);
            get(2).typeCheck(argType);
            break;
        }
        case IF: {
            get(1).typeCheck(Type.BOOLEAN);
            var argType = get(2).type();
            get(2).typeCheck(argType);
            get(3).typeCheck(argType);
            break;
        }
        default:
            throw new IllegalStateException(toString());
        }
    }

    public final void typeInfer(Type expected) {
        var map = new HashMap<TypeVariable, Type>();
        typeUnify(expected, map);
        typeAssign(map);
        typeCheck(expected);
    }

    private void typeUnify(Type expected, Map<TypeVariable, Type> map) {
        if (!type().unify(expected, map)) {
            throw new TypeException(this, type().replaceVars(map), expected.replaceVars(map));
        }
        switch (op()) {
        case ADD:
        case CEIL:
        case DIVIDE:
        case DIVIDE_EUCLIDEAN:
        case DIVIDE_FLOOR:
        case DIVIDE_TRUNCATE:
        case EQ:
        case FLOOR:
        case IS_INTEGER:
        case IS_RATIONAL:
        case LESS:
        case LESS_EQ:
        case MULTIPLY:
        case NEGATE:
        case REMAINDER_EUCLIDEAN:
        case REMAINDER_FLOOR:
        case REMAINDER_TRUNCATE:
        case ROUND:
        case SUBTRACT:
        case TO_INTEGER:
        case TO_RATIONAL:
        case TO_REAL:
        case TRUNCATE: {
            var argType = get(1).type();
            for (var i = 2; i < size(); i++) {
                get(i).typeUnify(argType, map);
            }
            break;
        }
        case ALL:
        case EXISTS:
            get(2).typeUnify(Type.BOOLEAN, map);
            break;
        case AND:
        case EQV:
        case NOT:
        case OR:
            for (var term : cdr()) {
                term.typeUnify(Type.BOOLEAN, map);
            }
            break;
        case ATOM:
            break;
        case CALL: {
            var funcType = get(0).type();
            if (funcType.size() == 0) {
                throw new TypeException(this, funcType);
            }
            if (funcType.size() != size()) {
                throw new TypeException(this, funcType);
            }
            for (var i = 1; i < size(); i++) {
                var param = funcType.get(i);
                get(i).typeUnify(param, map);
            }
            break;
        }
        default:
            throw new IllegalStateException(toString());
        }
    }

    public final boolean unequal(Term b) {
        if (isConstant() && b.isConstant()) {
            return !equals(b);
        }
        return false;
    }

    public boolean unify(Term b, Map<Variable, Term> map) {
        if (equals(b)) {
            return true;
        }
        if (b instanceof Variable) {
            return b.unify(this, map);
        }
        if (size() == 0) {
            return false;
        }
        if (!type().equals(b.type())) {
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

    public final Term unquantify() {
        var a = this;
        while (a.op() == Op.ALL) {
            a = a.get(2);
        }
        return a;
    }

    public final Set<Variable> variables() {
        Set<Variable> r = new LinkedHashSet<>();
        getVars(r);
        return r;
    }

    public final void walk(BiConsumer<Term, Integer> f) {
        walk(f, 0);
    }

    private void walk(BiConsumer<Term, Integer> f, int depth) {
        if (size() > 0) {
            if (isQuantifier()) {
                get(2).walk(f, depth + 1);
            } else {
                for (var a : cdr()) {
                    a.walk(f, depth + 1);
                }
            }
        }
        f.accept(this, depth);
    }

    public final void xml() {
        try {
            var writer = XMLOutputFactory.newFactory().createXMLStreamWriter(System.out);
            writer.writeStartDocument();
            writer.writeCharacters("\n");
            xml(writer, 0);
            writer.writeEndDocument();
            writer.close();
        } catch (XMLStreamException e) {
            throw new RuntimeException(e);
        }
    }

    public void xml(XMLStreamWriter writer, int depth) throws XMLStreamException {
        Util.startElement(writer, depth, tag().toString());
        writer.writeAttribute("class", getClass().getName());
        writer.writeAttribute("hash", Integer.toHexString(hashCode()));
        writer.writeAttribute("type", type().toString());
        writer.writeCharacters("\n");

        // Value
        if (size() == 0) {
            Util.text(writer, depth + 1, toString());
            writer.writeCharacters("\n");
        }

        // Contents
        for (var a : this) {
            a.xml(writer, depth + 1);
        }

        // End
        Util.endElement(writer, depth);
        writer.writeCharacters("\n");
    }

    public final Term xor(Term b) {
        return eqv(b).not();
    }

    private static class Compound extends Term {
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
        public boolean isConstant() {
            return false;
        }

        @Override
        public final Iterator<Term> iterator() {
            return new Iterator<>() {
                private int i;
                @Override
                public boolean hasNext() {
                    return i < size();
                }
                @Override
                public Term next() {
                    return get(i++);
                }
            };
        }

        @Override
        public final Op op() {
            return get(0).called();
        }

        @Override
        public Tag tag() {
            return Tag.LIST;
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
        private final Term[] data;

        CompoundList(Term... data) {
            this.data = data;
        }

        @Override
        public Term cdr() {
            if (size() == 0) {
                throw new IllegalStateException();
            }
            return new CompoundSlice(this, 1);
        }

        @Override
        public Term get(int i) {
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
        public Term cdr() {
            if (size() == 0) {
                throw new IllegalStateException();
            }
            return new CompoundSlice(list, offset + 1);
        }

        @Override
        public Term get(int i) {
            return list.get(offset + i);
        }

        @Override
        public int size() {
            return list.size() - offset;
        }
    }

    private static class Const extends Term {
        @Override
        public Term eval(Map<Variable, Term> map) {
            return this;
        }
    }

    private static class ConstInteger extends Const {
        private final BigInteger value;

        ConstInteger(BigInteger value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if ((o == null) || (getClass() != o.getClass())) {
                return false;
            }
            ConstInteger terms = (ConstInteger) o;
            return Objects.equals(value, terms.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(value);
        }

        @Override
        public BigInteger integerValue() {
            return value;
        }

        @Override
        public boolean isOne() {
            return value.equals(BigInteger.ONE);
        }

        @Override
        public boolean isZero() {
            return value.signum() == 0;
        }

        @Override
        public Tag tag() {
            return Tag.CONST_INTEGER;
        }

        @Override
        public String toString() {
            return value.toString();
        }

        @Override
        public Type type() {
            return INTEGER;
        }
    }

    private static class ConstRational extends Const {
        private final BigRational value;

        ConstRational(BigRational value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if ((o == null) || (getClass() != o.getClass())) {
                return false;
            }
            ConstRational terms = (ConstRational) o;
            return Objects.equals(value, terms.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(value);
        }

        @Override
        public boolean isOne() {
            return value.equals(BigRational.ONE);
        }

        @Override
        public boolean isZero() {
            return value.signum() == 0;
        }

        @Override
        public BigRational rationalValue() {
            return value;
        }

        @Override
        public Tag tag() {
            return Tag.CONST_RATIONAL;
        }

        @Override
        public String toString() {
            return value.toString();
        }

        @Override
        public Type type() {
            return RATIONAL;
        }
    }

    private static class ConstReal extends Const {
        private final double value;

        ConstReal(double value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if ((o == null) || (getClass() != o.getClass())) {
                return false;
            }
            ConstReal terms = (ConstReal) o;
            return Double.compare(terms.value, value) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(value);
        }

        @Override
        public boolean isOne() {
            return value == 1.0;
        }

        @Override
        public boolean isZero() {
            return value == 0.0;
        }

        @Override
        public double realValue() {
            return value;
        }

        @Override
        public Tag tag() {
            return Tag.CONST_REAL;
        }

        @Override
        public String toString() {
            return Double.toString(value);
        }

        @Override
        public Type type() {
            return REAL;
        }
    }

    private static abstract class Op2 {
        abstract Term apply(BigInteger a, BigInteger b);

        abstract Term apply(BigRational a, BigRational b);

        abstract Term apply(double a, double b);
    }
}
