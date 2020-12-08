package prover;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.WeakHashMap;

import org.junit.Test;

import static org.junit.Assert.*;

public class SubsumptionTest {
    private Term random(Function[] predicates, Function[] functions, Term[] atoms, Random random) {
        var depth = random.nextInt(5);
        if (random.nextInt(3) == 0) {
            var a = random(functions, atoms, depth, random);
            var b = random(functions, atoms, depth, random);
            switch (random.nextInt(5)) {
            case 0:
            case 1:
            case 2:
                return a.eq(b);
            case 3:
                return a.less(b);
            case 4:
                return a.lessEq(b);
            }
        }
        var f = predicates[random.nextInt(predicates.length)];
        var n = f.arity();
        if (n > 0) {
            var args = new Term[n];
            for (var i = 0; i < args.length; i++) {
                args[i] = random(functions, atoms, depth, random);
            }
            return f.call(args);
        }
        return f;
    }

    private Term random(Function[] functions, Term[] atoms, int depth, Random random) {
        if (depth > 0) {
            if (random.nextInt(10) == 0) {
                var a = random(functions, atoms, depth - 1, random);
                var b = random(functions, atoms, depth - 1, random);
                switch (random.nextInt(3)) {
                case 0:
                    return a.add(b);
                case 1:
                    return a.subtract(b);
                case 2:
                    return a.multiply(b);
                }
            }
            var f = functions[random.nextInt(functions.length)];
            var n = f.arity();
            if (n > 0) {
                var args = new Term[n];
                for (var i = 0; i < args.length; i++) {
                    args[i] = random(functions, atoms, depth - 1, random);
                }
                return f.call(args);
            }
            return f;
        }
        return atoms[random.nextInt(atoms.length)];
    }

    private Clause randomClause(Function[] predicates, Function[] functions, Term[] atoms, Random random) {
        var n = 1 + random.nextInt(5);
        var negative = new ArrayList<Term>(n);
        for (var i = 0; i < n; i++) {
            negative.add(random(predicates, functions, atoms, random));
        }
        n = 1 + random.nextInt(5);
        var positive = new ArrayList<Term>(n);
        for (var i = 0; i < n; i++) {
            positive.add(random(predicates, functions, atoms, random));
        }
        return new ClauseInput(negative, positive, null, null);
    }

    private List<Clause> randomClauses(int n) {

        // Predicates
        var predicates = new ArrayList<Function>();
        predicates.add(new Function(Type.BOOLEAN, "p"));
        predicates.add(new Function(Type.of(Type.BOOLEAN, Type.INTEGER), "p1"));
        predicates.add(new Function(Type.of(Type.BOOLEAN, Type.INTEGER, Type.INTEGER), "p2"));
        predicates.add(new Function(Type.BOOLEAN, "q"));
        predicates.add(new Function(Type.of(Type.BOOLEAN, Type.INTEGER), "q1"));
        predicates.add(new Function(Type.of(Type.BOOLEAN, Type.INTEGER, Type.INTEGER), "q2"));
        var predicates1 = predicates.toArray(new Function[0]);

        // Functions
        var functions = new ArrayList<Function>();
        functions.add(new Function(Type.of(Type.INTEGER, Type.INTEGER), "a1"));
        functions.add(new Function(Type.of(Type.INTEGER, Type.INTEGER, Type.INTEGER), "a2"));
        functions.add(new Function(Type.of(Type.INTEGER, Type.INTEGER), "b1"));
        functions.add(new Function(Type.of(Type.INTEGER, Type.INTEGER, Type.INTEGER), "b2"));
        var funcs1 = functions.toArray(new Function[0]);

        // Atoms
        var atoms = new ArrayList<Term>();
        atoms.add(Term.of(0));
        atoms.add(Term.of(1));
        atoms.add(new Variable(Type.INTEGER));
        atoms.add(new Variable(Type.INTEGER));
        atoms.add(new Variable(Type.INTEGER));
        atoms.add(new Variable(Type.INTEGER));
        atoms.add(new Function(Type.INTEGER, "a"));
        atoms.add(new Function(Type.INTEGER, "b"));
        var atoms1 = atoms.toArray(new Term[0]);

        // Clauses
        var random = new Random(0);
        var r = new ArrayList<Clause>();
        for (var i = 0; i < n; i++) {
            var c = randomClause(predicates1, funcs1, atoms1, random);
            r.add(c);
        }
        return r;
    }

    @Test
    public void randomTest() {

        // For a more thorough but slower test
        // increase n to e.g. 10000
        var clauses = randomClauses(300);
        var subsumption = new Subsumption(clauses);

        // Slow way
        var clauses1 = new ArrayList<Clause>();
        var subsumed1 = new WeakHashMap<Clause, Boolean>();
        loop:
        for (var d : clauses) {
            for (var c : clauses1) {
                if (subsumption.subsumes(c, d)) {
                    continue loop;
                }
            }
            for (var c : clauses1) {
                if (subsumption.subsumes(d, c)) {
                    subsumed1.put(c, true);
                }
            }
            clauses1.add(d);
        }

        // Fast way
        var clauses2 = new ArrayList<Clause>();
        for (var d : clauses) {
            if (!subsumption.add(d)) {
                continue;
            }
            clauses2.add(d);
        }

        // Compare
        assertEquals(clauses1, clauses2);
        for (var i = 0; i < clauses1.size(); i++) {
            assertEquals(subsumed1.containsKey(clauses1.get(i)), subsumption.subsumed(clauses2.get(i)));
        }
    }

    @Test
    public void subsumes() {
        var a = new Function(Type.INTEGER, "a");
        var a1 = new Function(Type.of(Type.INTEGER, Type.INTEGER), "a1");
        var b = new Function(Type.INTEGER, "b");
        var p = new Function(Type.BOOLEAN, "p");
        var p1 = new Function(Type.of(Type.BOOLEAN, Type.INTEGER), "p1");
        var p2 = new Function(Type.of(Type.BOOLEAN, Type.INTEGER, Type.INTEGER), "p2");
        var q = new Function(Type.BOOLEAN, "q");
        var q1 = new Function(Type.of(Type.BOOLEAN, Type.INTEGER), "q1");
        var q2 = new Function(Type.of(Type.BOOLEAN, Type.INTEGER, Type.INTEGER), "q2");
        var x = new Variable(Type.INTEGER);
        var y = new Variable(Type.INTEGER);
        var negative = new ArrayList<Term>();
        var positive = new ArrayList<Term>();
        Clause c, d;
        var subsumption = new Subsumption(new ArrayList<>());

        // false <= false
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));

        // false <= p
        negative.clear();
        positive.clear();
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p);
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p <= p
        negative.clear();
        positive.clear();
        positive.add(p);
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p);
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));

        // !p <= !p
        negative.clear();
        negative.add(p);
        positive.clear();
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        negative.add(p);
        positive.clear();
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));

        // p <= p | p
        negative.clear();
        positive.clear();
        positive.add(p);
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p);
        positive.add(p);
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p !<= !p
        negative.clear();
        positive.clear();
        positive.add(p);
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        negative.add(p);
        positive.clear();
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertFalse(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p | q <= q | p
        negative.clear();
        positive.clear();
        positive.add(p);
        positive.add(q);
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(q);
        positive.add(p);
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // p | q <= p | q | p
        negative.clear();
        positive.clear();
        positive.add(p);
        positive.add(q);
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p);
        positive.add(q);
        positive.add(p);
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
        negative.clear();
        positive.clear();
        positive.add(p1.call(a));
        positive.add(p1.call(b));
        positive.add(q1.call(a));
        positive.add(q1.call(b));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(a));
        positive.add(q1.call(a));
        positive.add(p1.call(b));
        positive.add(q1.call(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // p(6,7) | p(4,5) <= q(6,7) | q(4,5) | p(0,1) | p(2,3) | p(4,4) | p(4,5) | p(6,6) | p(6,7)
        negative.clear();
        positive.clear();
        positive.add(p2.call(Term.of(6), Term.of(7)));
        positive.add(p2.call(Term.of(4), Term.of(5)));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(q2.call(Term.of(6), Term.of(7)));
        positive.add(q2.call(Term.of(4), Term.of(5)));
        positive.add(p2.call(Term.of(0), Term.of(1)));
        positive.add(p2.call(Term.of(2), Term.of(3)));
        positive.add(p2.call(Term.of(4), Term.of(4)));
        positive.add(p2.call(Term.of(4), Term.of(5)));
        positive.add(p2.call(Term.of(6), Term.of(6)));
        positive.add(p2.call(Term.of(6), Term.of(7)));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(x,y) <= p(a,b)
        negative.clear();
        positive.clear();
        positive.add(p2.call(x, y));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p2.call(a, b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(x,x) !<= p(a,b)
        negative.clear();
        positive.clear();
        positive.add(p2.call(x, x));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p2.call(a, b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertFalse(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(x) <= p(y)
        negative.clear();
        positive.clear();
        positive.add(p1.call(x));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(y));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
        negative.clear();
        positive.clear();
        positive.add(p1.call(x));
        positive.add(p1.call(a1.call(x)));
        positive.add(p1.call(a1.call(a1.call(x))));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(y));
        positive.add(p1.call(a1.call(y)));
        positive.add(p1.call(a1.call(a1.call(y))));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // p(x) | p(a) <= p(a) | p(b)
        negative.clear();
        positive.clear();
        positive.add(p1.call(x));
        positive.add(p1.call(a));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(a));
        positive.add(p1.call(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(x) | p(a(x)) <= p(a(y)) | p(y)
        negative.clear();
        positive.clear();
        positive.add(p1.call(x));
        positive.add(p1.call(a1.call(x)));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(a1.call(y)));
        positive.add(p1.call(y));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
        negative.clear();
        positive.clear();
        positive.add(p1.call(x));
        positive.add(p1.call(a1.call(x)));
        positive.add(p1.call(a1.call(a1.call(x))));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p1.call(a1.call(a1.call(y))));
        positive.add(p1.call(a1.call(y)));
        positive.add(p1.call(y));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertTrue(subsumption.subsumes(d, c));

        // (a = x) <= (a = b)
        negative.clear();
        positive.clear();
        positive.add(a.eq(x));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(a.eq(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // (x = a) <= (a = b)
        negative.clear();
        positive.clear();
        positive.add(x.eq(a));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(a.eq(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
        negative.clear();
        negative.add(p1.call(y));
        negative.add(p1.call(x));
        positive.clear();
        positive.add(q1.call(x));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        negative.add(p1.call(a));
        negative.add(p1.call(b));
        positive.clear();
        positive.add(q1.call(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
        negative.clear();
        negative.add(p1.call(x));
        negative.add(p1.call(y));
        positive.clear();
        positive.add(q1.call(x));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        negative.add(p1.call(a));
        negative.add(p1.call(b));
        positive.clear();
        positive.add(q1.call(b));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertTrue(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // (x = a) | (1 = y) !<= (1 = a) | (x = 0)
        negative.clear();
        positive.clear();
        positive.add(x.eq(a));
        positive.add(Term.of(1).eq(y));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(Term.of(1).eq(a));
        positive.add(x.eq(Term.of(0)));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertFalse(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));

        // p(x,a(x)) !<= p(a(y),a(y))
        negative.clear();
        positive.clear();
        positive.add(p2.call(x, a1.call(x)));
        c = new ClauseInput(negative, positive, null, null);
        c.typeCheck();
        negative.clear();
        positive.clear();
        positive.add(p2.call(a1.call(y), a1.call(y)));
        d = new ClauseInput(negative, positive, null, null);
        d.typeCheck();
        assertFalse(subsumption.subsumes(c, d));
        assertFalse(subsumption.subsumes(d, c));
    }
}
