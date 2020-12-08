package prover;

import java.math.BigInteger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;

public class Search {
    private static final int tries = 100000;
    public static Random random = new Random(0);
    public static List<Term> atoms = new ArrayList<>();

    private static Type domain() {
        return atoms.get(0).type();
    }

    public static Term hillClimb(Type type, Function<Term, Double> cost) {
        var a = random(type, 3);
        var c = cost.apply(a);
        for (var i = 0; i < tries; i++) {
            var a1 = mutate(a);
            if (a1.deepSize() > 100) {
                continue;
            }
            var c1 = cost.apply(a1);
            if (c1 < c) {
                a = a1;
                c = c1;
            }
        }
        a = a.simplify();
        return a;
    }

    public static Term mutate(Term a) {
        var n = a.deepSize();
        var i = random.nextInt(n);
        var b = a.deepGet(i);
        var t = b.type();
        return a.deepSplice(i, random(t, random.nextInt(3)));
    }

    private static Op random(Op[] values) {
        var i = random.nextInt(values.length);
        return values[i];
    }

    public static Term random(Type type, int depth) {
        switch (type.kind()) {
        case BOOLEAN:
            return randomBoolean(depth);
        case INTEGER:
            return randomInteger(depth);
        case REAL:
            return randomReal(depth);
        }
        throw new IllegalArgumentException(type.toString());
    }

    public static Term random(Type type, Predicate<Term> accept) {
        for (var i = 0; i < tries; i++) {
            var a = random(type, random.nextInt(4));
            if (accept.test(a)) {
                return a;
            }
        }
        return null;
    }

    private static Term randomAtom() {
        return atoms.get(random.nextInt(atoms.size()));
    }

    public static Term randomBoolean(int depth) {
        if (depth == 0) {
            if ((domain() == Type.BOOLEAN) && (random.nextInt(2) == 0)) {
                return randomAtom();
            }
            return Term.of(random.nextBoolean());
        }
        depth--;
        var op = random(new Op[] {
            Op.AND, Op.EQ, Op.EQV, Op.LESS, Op.LESS_EQ, Op.NOT, Op.OR
        });
        switch (op) {
        case AND:
            return randomBoolean(depth).and(randomBoolean(depth));
        case EQ: {
            var type = domain();
            return random(type, depth).eq(random(type, depth));
        }
        case EQV:
            return randomBoolean(depth).eqv(randomBoolean(depth));
        case LESS: {
            var type = domain();
            return random(type, depth).less(random(type, depth));
        }
        case LESS_EQ: {
            var type = domain();
            return random(type, depth).lessEq(random(type, depth));
        }
        case NOT:
            return randomBoolean(depth).not();
        case OR:
            return randomBoolean(depth).or(randomBoolean(depth));
        }
        throw new IllegalStateException(op.toString());
    }

    public static Term randomInteger(int depth) {
        if (depth == 0) {
            if ((domain() == Type.INTEGER) && (random.nextInt(2) == 0)) {
                return randomAtom();
            }
            return Term.of(BigInteger.valueOf(random.nextInt(10)));
        }
        depth--;
        var op = random(new Op[] { Op.ADD, Op.IF, Op.MULTIPLY, Op.NEGATE, Op.SUBTRACT });
        switch (op) {
        case ADD:
            return randomInteger(depth).add(randomInteger(depth));
        case IF:
            return Term.of(Term.IF, randomBoolean(depth), randomInteger(depth), randomInteger(depth));
        case MULTIPLY:
            return randomInteger(depth).multiply(randomInteger(depth));
        case NEGATE:
            return randomInteger(depth).negate();
        case SUBTRACT:
            return randomInteger(depth).subtract(randomInteger(depth));
        }
        throw new IllegalStateException(op.toString());
    }

    public static Term randomReal(int depth) {
        if (depth == 0) {
            if ((domain() == Type.REAL) && (random.nextInt(2) == 0)) {
                return randomAtom();
            }
            return Term.of(random.nextDouble());
        }
        depth--;
        var op = random(new Op[] { Op.ADD, Op.IF, Op.MULTIPLY, Op.NEGATE, Op.SUBTRACT });
        switch (op) {
        case ADD:
            return randomReal(depth).add(randomReal(depth));
        case IF:
            return Term.of(Term.IF, randomBoolean(depth), randomReal(depth), randomReal(depth));
        case MULTIPLY:
            return randomReal(depth).multiply(randomReal(depth));
        case NEGATE:
            return randomReal(depth).negate();
        case SUBTRACT:
            return randomReal(depth).subtract(randomReal(depth));
        }
        throw new IllegalStateException(op.toString());
    }

    public static void test() {
        var x = new Variable(Type.INTEGER);
        atoms.add(x);
        var a = Search.hillClimb(Type.INTEGER,
                    b -> {
                        double total = 0;
                        for (var i = 0; i < 100; i++) {
                            var answer = Math.sqrt(i);
                            var map = new HashMap<Variable, Term>();
                            map.put(x, Term.of(i));
                            var r = b.eval(map).integerValue();
                            var err = Math.abs(r.doubleValue() - answer);
                            total += err;
                        }
                        return total;
                    });
        a.println();
    }
}
