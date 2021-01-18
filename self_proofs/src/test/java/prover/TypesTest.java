package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.List;
import org.junit.Test;

public class TypesTest {
  @Test
  public void typeof() {
    assertEquals(Types.typeof(false), Symbol.BOOLEAN);
    assertEquals(Types.typeof(true), Symbol.BOOLEAN);
    assertEquals(Types.typeof(BigInteger.ZERO), Symbol.INTEGER);
    assertEquals(Types.typeof(BigRational.ZERO), Symbol.RATIONAL);
    assertEquals(Types.typeof(new Variable(Symbol.INDIVIDUAL)), Symbol.INDIVIDUAL);
  }

  @Test
  public void inferTypes() {
    // p
    var p = new Func(new Variable(null), "p");
    var formula = new Formula(p, Inference.AXIOM);
    Types.inferTypes(List.of(formula), List.of());
    assertEquals(Types.typeof(p), Symbol.BOOLEAN);

    // !p
    p = new Func(new Variable(null), "p");
    formula = new Formula(List.of(Symbol.NOT, p), Inference.AXIOM);
    Types.inferTypes(List.of(formula), List.of());
    assertEquals(Types.typeof(p), Symbol.BOOLEAN);

    // !!p
    p = new Func(new Variable(null), "p");
    formula = new Formula(List.of(Symbol.NOT, List.of(Symbol.NOT, p)), Inference.AXIOM);
    Types.inferTypes(List.of(formula), List.of());
    assertEquals(Types.typeof(p), Symbol.BOOLEAN);

    // p | q
    p = new Func(new Variable(null), "p");
    var q = new Func(new Variable(null), "q");
    formula = new Formula(List.of(Symbol.OR, p, q), Inference.AXIOM);
    Types.inferTypes(List.of(formula), List.of());
    assertEquals(Types.typeof(p), Symbol.BOOLEAN);
    assertEquals(Types.typeof(q), Symbol.BOOLEAN);

    // !!((!p | q(r)) & (q(s) | p))
    p = new Func(new Variable(null), "p");
    q = new Func(List.of(new Variable(null), new Variable(null)), "q");
    var r = new Func(new Variable(null), "r");
    var s = new Func(new Variable(null), "s");
    formula =
        new Formula(
            List.of(
                Symbol.NOT,
                List.of(
                    Symbol.NOT,
                    List.of(
                        Symbol.AND,
                        List.of(Symbol.OR, List.of(Symbol.NOT, p), List.of(q, r)),
                        List.of(Symbol.OR, List.of(q, s), p)))),
            Inference.AXIOM);
    Types.inferTypes(List.of(formula), List.of());
    assertEquals(Types.typeof(p), Symbol.BOOLEAN);
    assertEquals(Types.typeof(q), List.of(Symbol.BOOLEAN, Symbol.INDIVIDUAL));
    assertEquals(Types.typeof(r), Symbol.INDIVIDUAL);
    assertEquals(Types.typeof(s), Symbol.INDIVIDUAL);
  }
}
