package prover;

import java.util.*;

public final class DPLL extends Thread {
    private Collection<Clause> input;
    public SZS szs;

    public DPLL(Collection<Clause> clauses) {
        input = clauses;
    }

    private static boolean isFalse(Collection<Clause> clauses) {
        return (clauses.size() == 1) && clauses.iterator().next().isFalse();
    }

    private static boolean isTrue(Collection<Clause> clauses) {
        return clauses.isEmpty();
    }

    private static Set<Term> literals(Collection<Clause> clauses) {
        var r = new HashSet<Term>();
        for (var c : clauses) {
            Collections.addAll(r, c.literals);
        }
        return r;
    }

    private static boolean occursNegative(Term a, Collection<Clause> clauses) {
        for (var c : clauses) {
            for (var i = 0; i < c.negativeSize; i++) {
                if (c.literals[i].equals(a)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean occursPositive(Term a, Collection<Clause> clauses) {
        for (var c : clauses) {
            for (var i = c.negativeSize; i < c.literals.length; i++) {
                if (c.literals[i].equals(a)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static Clause replace(Clause c, Map<Term, Term> map) {
        var negative = new ArrayList<Term>(c.negativeSize);
        for (var i = 0; i < c.negativeSize; i++) {
            negative.add(c.literals[i].replace(map));
        }
        var positive = new ArrayList<Term>(c.positiveSize());
        for (var i = c.negativeSize; i < c.literals.length; i++) {
            positive.add(c.literals[i].replace(map));
        }

        // If we find a satisfying assignment,
        // don't need to track derivations
        return new Clause(negative, positive);
    }

    private static Set<Clause> replace(Collection<Clause> clauses, Map<Term, Term> map) {
        var r = new HashSet<Clause>();
        for (var c : clauses) {
            c = replace(c, map);
            if (c.isFalse()) {
                r = new HashSet<>();
                r.add(c);
                break;
            }
            if (c.isTrue()) {
                continue;
            }
            r.add(c);
        }
        return r;
    }

    @Override
    public void run() {
        try {
            szs = sat(input, new HashMap<>())
                  ? SZS.Satisfiable
                  : SZS.Unsatisfiable;
        } catch (InterruptedException e) {
            szs = SZS.Timeout;
        }
    }

    private boolean sat(Collection<Clause> clauses, Map<Term, Term> map) throws InterruptedException {

        // Timeout
        if (Thread.interrupted()) {
            throw new InterruptedException();
        }

        // Evaluate previous literal assignment
        clauses = replace(clauses, map);

        // Solved?
        if (isFalse(clauses)) {
            return false;
        }
        if (isTrue(clauses)) {
            return true;
        }

        // Unit clause
        for (var c : clauses) {
            if (c.literals.length == 1) {
                map.put(c.literals[0], Term.of(c.negativeSize == 0));
                return sat(clauses, map);
            }
        }

        // Pure literal
        var literals = literals(clauses);
        for (var a : literals) {
            if (!occursNegative(a, clauses)) {
                map.put(a, Term.TRUE);
                return sat(clauses, map);
            }
            if (!occursPositive(a, clauses)) {
                map.put(a, Term.FALSE);
                return sat(clauses, map);
            }
        }

        // Choose a literal
        var a = literals.iterator().next();

        // Try assigning false
        var old = new HashMap<>(map);
        map.put(a, Term.FALSE);
        if (sat(clauses, map)) {
            return true;
        }

        // Try assigning true
        map = old;
        map.put(a, Term.TRUE);
        return sat(clauses, map);
    }
}
