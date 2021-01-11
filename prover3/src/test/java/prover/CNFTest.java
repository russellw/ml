package prover;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

public class CNFTest {
  private static void assertEql(Clause c, Object... q) {
    var negativeIndex = 0;
    var positiveIndex = 0;
    for (var a : q) {
      if (Etc.head(a) == Symbol.NOT) assertEquals(c.negative()[negativeIndex++], ((List) a).get(1));
      else assertEquals(c.positive()[positiveIndex++], a);
    }
  }

  @Test
  public void convert() {
    Formula formula;
    List<Clause> clauses;

    // false
    formula = new Formula(false, Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertTrue(clauses.get(0).isFalse());

    // true
    formula = new Formula(true, Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // !false
    formula = new Formula(List.of(Symbol.NOT, false), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // !true
    formula = new Formula(List.of(Symbol.NOT, true), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertTrue(clauses.get(0).isFalse());

    // false & false
    formula = new Formula(List.of(Symbol.AND, false, false), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 2);
    assertTrue(clauses.get(0).isFalse());
    assertTrue(clauses.get(1).isFalse());

    // false & true
    formula = new Formula(List.of(Symbol.AND, false, true), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertTrue(clauses.get(0).isFalse());

    // true & false
    formula = new Formula(List.of(Symbol.AND, true, false), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertTrue(clauses.get(0).isFalse());

    // true & true
    formula = new Formula(List.of(Symbol.AND, true, true), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // false | false
    formula = new Formula(List.of(Symbol.OR, false, false), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertTrue(clauses.get(0).isFalse());

    // false | true
    formula = new Formula(List.of(Symbol.OR, false, true), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // true | false
    formula = new Formula(List.of(Symbol.OR, true, false), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // true | true
    formula = new Formula(List.of(Symbol.OR, true, true), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 0);

    // a & b
    var a = new Func(Symbol.BOOLEAN, "a");
    var b = new Func(Symbol.BOOLEAN, "b");
    formula = new Formula(List.of(Symbol.AND, a, b), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 2);
    assertEql(clauses.get(0), a);
    assertEql(clauses.get(1), b);

    // a | b
    formula = new Formula(List.of(Symbol.OR, a, b), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertEql(clauses.get(0), a, b);

    // !(a & b)
    formula = new Formula(List.of(Symbol.NOT, List.of(Symbol.AND, a, b)), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 1);
    assertEql(clauses.get(0), List.of(Symbol.NOT, a), List.of(Symbol.NOT, b));

    // !(a | b)
    formula = new Formula(List.of(Symbol.NOT, List.of(Symbol.OR, a, b)), Inference.AXIOM);
    clauses = convert1(formula);
    assertEquals(clauses.size(), 2);
    assertEql(clauses.get(0), List.of(Symbol.NOT, a));
    assertEql(clauses.get(1), List.of(Symbol.NOT, b));
  }

  private static List<Clause> convert1(Formula formula) {
    var clauses = new ArrayList<Clause>();
    CNF.convert(Collections.singletonList(formula), clauses);
    return clauses;
  }
}
