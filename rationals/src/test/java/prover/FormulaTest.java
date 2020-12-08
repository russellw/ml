package prover;

import static org.junit.Assert.*;

import org.junit.Test;

public class FormulaTest {
  @Test
  public void status() {
    var a = new Function(Type.BOOLEAN, "a");
    var b = new Function(Type.BOOLEAN, "b");

    // Axiom
    var axiom = new FormulaTerm(a, null);
    assertEquals(axiom.szs(), SZS.LogicalData);

    // Equivalent
    var eqv = new FormulaTermFrom(a, axiom);
    assertEquals(eqv.szs(), SZS.Equivalent);

    // Conjecture
    var conjecture = new FormulaTermInputConjecture(a, null, null);
    assertEquals(conjecture.szs(), SZS.LogicalData);

    // Negated conjecture
    var negatedConjecture = new FormulaTermFrom(a.not(), conjecture);
    assertEquals(negatedConjecture.szs(), SZS.CounterEquivalent);

    // Equisatisfiable
    // refers to a new symbol
    var esa = new FormulaTermFrom(a.or(b), axiom);
    assertEquals(esa.szs(), SZS.EquiSatisfiable);

    // Theorem
    // cannot check it is actually a theorem
    // that is the job of verify mode and a second prover
    var theorem = new FormulaTermFrom(a.and(Term.TRUE), axiom);
    assertEquals(theorem.szs(), SZS.Theorem);
  }
}
