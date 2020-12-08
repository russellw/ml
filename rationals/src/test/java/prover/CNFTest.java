package prover;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import static org.junit.Assert.*;

public class CNFTest {
    private static void assertEql(Clause c, Term... q) {
        var negativeIndex = 0;
        var positiveIndex = 0;
        for (var a : q) {
            if (a.op() == Op.NOT) {
                assertEquals(c.negative()[negativeIndex++], a.get(1));
            } else {
                assertEquals(c.positive()[positiveIndex++], a);
            }
        }
    }

    @Test
    public void convert() {
        Formula formula;
        List<Clause> clauses;

        // false
        formula = new FormulaTermInput(Term.FALSE, null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertTrue(clauses.get(0).isFalse());

        // true
        formula = new FormulaTermInput(Term.TRUE, null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // !false
        formula = new FormulaTermInput(Term.FALSE.not(), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // !true
        formula = new FormulaTermInput(Term.TRUE.not(), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertTrue(clauses.get(0).isFalse());

        // false & false
        formula = new FormulaTermInput(Term.FALSE.and(Term.FALSE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 2);
        assertTrue(clauses.get(0).isFalse());
        assertTrue(clauses.get(1).isFalse());

        // false & true
        formula = new FormulaTermInput(Term.FALSE.and(Term.TRUE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertTrue(clauses.get(0).isFalse());

        // true & false
        formula = new FormulaTermInput(Term.TRUE.and(Term.FALSE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertTrue(clauses.get(0).isFalse());

        // true & true
        formula = new FormulaTermInput(Term.TRUE.and(Term.TRUE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // false | false
        formula = new FormulaTermInput(Term.FALSE.or(Term.FALSE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertTrue(clauses.get(0).isFalse());

        // false | true
        formula = new FormulaTermInput(Term.FALSE.or(Term.TRUE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // true | false
        formula = new FormulaTermInput(Term.TRUE.or(Term.FALSE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // true | true
        formula = new FormulaTermInput(Term.TRUE.or(Term.TRUE), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 0);

        // a & b
        var a = new Function(Type.BOOLEAN, "a");
        var b = new Function(Type.BOOLEAN, "b");
        formula = new FormulaTermInput(a.and(b), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 2);
        assertEql(clauses.get(0), a);
        assertEql(clauses.get(1), b);

        // a | b
        formula = new FormulaTermInput(a.or(b), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertEql(clauses.get(0), a, b);

        // !(a & b)
        formula = new FormulaTermInput(a.and(b).not(), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 1);
        assertEql(clauses.get(0), a.not(), b.not());

        // !(a | b)
        formula = new FormulaTermInput(a.or(b).not(), null, null);
        clauses = convert1(formula);
        assertEquals(clauses.size(), 2);
        assertEql(clauses.get(0), a.not());
        assertEql(clauses.get(1), b.not());
    }

    private static List<Clause> convert1(Formula formula) {
        var clauses = new ArrayList<Clause>();
        new CNF(Collections.singletonList(formula), clauses);
        return clauses;
    }
}
