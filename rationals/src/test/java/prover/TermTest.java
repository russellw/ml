package prover;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

public class TermTest {
  private void assertEval(Term a, Term b) {
    assertEquals(a.eval(new HashMap<>()), b);
  }

  @Test
  public void deepGet() {
    assertEquals(Term.of(1.0).deepGet(0), Term.of(1.0));
    assertEquals(
        Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepGet(0),
        Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)));
    assertEquals(Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepGet(1), Term.of(1.0));
    assertEquals(Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepGet(2), Term.of(2.0));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(0),
        Term.of(
            Term.ADD,
            Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
            Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0))));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(1),
        Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(2),
        Term.of(2.0));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(3),
        Term.of(3.0));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(4),
        Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(5),
        Term.of(5.0));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.SUBTRACT, Term.of(2.0), Term.of(3.0)),
                Term.of(Term.MULTIPLY, Term.of(5.0), Term.of(6.0)))
            .deepGet(6),
        Term.of(6.0));
  }

  @Test
  public void deepSize() {
    assertEquals(Term.of(1.0).deepSize(), 1);
    assertEquals(Term.of(Term.NOT, Term.TRUE).deepSize(), 2);
    assertEquals(Term.of(Term.AND, Term.TRUE, Term.FALSE).deepSize(), 3);
  }

  @Test
  public void deepSplice() {
    assertEquals(Term.of(1.0).deepSplice(0, Term.of(2.0)), Term.of(2.0));
    assertEquals(
        Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepSplice(0, Term.of(9.0)), Term.of(9.0));
    assertEquals(
        Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepSplice(1, Term.of(9.0)),
        Term.of(Term.ADD, Term.of(9.0), Term.of(2.0)));
    assertEquals(
        Term.of(Term.ADD, Term.of(1.0), Term.of(2.0)).deepSplice(2, Term.of(9.0)),
        Term.of(Term.ADD, Term.of(1.0), Term.of(9.0)));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
                Term.of(Term.ADD, Term.of(2.0), Term.of(2.0)))
            .deepSplice(2, Term.of(9.0)),
        Term.of(
            Term.ADD,
            Term.of(Term.ADD, Term.of(9.0), Term.of(1.0)),
            Term.of(Term.ADD, Term.of(2.0), Term.of(2.0))));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
                Term.of(Term.ADD, Term.of(2.0), Term.of(2.0)))
            .deepSplice(3, Term.of(9.0)),
        Term.of(
            Term.ADD,
            Term.of(Term.ADD, Term.of(1.0), Term.of(9.0)),
            Term.of(Term.ADD, Term.of(2.0), Term.of(2.0))));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
                Term.of(Term.ADD, Term.of(2.0), Term.of(2.0)))
            .deepSplice(5, Term.of(9.0)),
        Term.of(
            Term.ADD,
            Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
            Term.of(Term.ADD, Term.of(9.0), Term.of(2.0))));
    assertEquals(
        Term.of(
                Term.ADD,
                Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
                Term.of(Term.ADD, Term.of(2.0), Term.of(2.0)))
            .deepSplice(6, Term.of(9.0)),
        Term.of(
            Term.ADD,
            Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)),
            Term.of(Term.ADD, Term.of(2.0), Term.of(9.0))));
  }

  @Test
  public void eval() {
    var x = new Variable(Type.REAL);
    var map = new HashMap<Variable, Term>();
    map.put(x, Term.of(1.0));

    // Boolean operators
    assertEquals(Term.of(1).eval(map), Term.of(1));
    assertEquals(Term.TRUE.eval(map), Term.TRUE);
    assertEquals(Term.of(Term.NOT, Term.FALSE).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.AND).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.AND, Term.TRUE).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.AND, Term.TRUE, Term.TRUE).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.AND, Term.TRUE, Term.TRUE, Term.TRUE).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.AND, Term.TRUE, Term.TRUE, Term.FALSE).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.OR).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.OR, Term.FALSE).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.OR, Term.FALSE, Term.FALSE).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.OR, Term.FALSE, Term.FALSE, Term.FALSE).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.OR, Term.FALSE, Term.FALSE, Term.TRUE).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.ADD, Term.of(0), Term.of(1)).eval(map), Term.of(1));

    // Variables
    assertEquals(x.eval(map), Term.of(1.0));

    // Add integers and reals
    assertEquals(Term.of(1).add(Term.of(2)).eval(map), Term.of(3));
    assertEquals(Term.of(1).add(Term.of(2.0)).eval(map), Term.of(3.0));
    assertEquals(Term.of(1.0).add(Term.of(2)).eval(map), Term.of(3.0));
    assertEquals(Term.of(1.0).add(Term.of(2.0)).eval(map), Term.of(3.0));

    // Add integers and rationals
    assertEquals(Term.of(1).add(Term.of(2)).eval(map), Term.of(3));
    assertEquals(
        Term.of(1).add(Term.of(BigRational.of("2/1"))).eval(map), Term.of(BigRational.of("3/1")));
    assertEquals(
        Term.of(BigRational.of("1/1")).add(Term.of(2)).eval(map), Term.of(BigRational.of("3/1")));
    assertEquals(
        Term.of(BigRational.of("1/1")).add(Term.of(BigRational.of("2/1"))).eval(map),
        Term.of(BigRational.of("3/1")));

    // Add rationals and reaLs
    assertEquals(Term.of(1.0).add(Term.of(2.0)).eval(map), Term.of(3.0));
    assertEquals(Term.of(1.0).add(Term.of(BigRational.of("2/1"))).eval(map), Term.of(3.0));
    assertEquals(Term.of(BigRational.of("1/1")).add(Term.of(2.0)).eval(map), Term.of(3.0));
    assertEquals(
        Term.of(BigRational.of("1/1")).add(Term.of(BigRational.of("2/1"))).eval(map),
        Term.of(BigRational.of("3/1")));

    // Different types
    assertNotEquals(Term.of(1), Term.of(1.0));
    assertNotEquals(Term.of(1), Term.of(BigRational.of(1)));
    assertNotEquals(Term.of(1.0), Term.of(BigRational.of(1)));

    // Different values
    assertNotEquals(Term.of(1), Term.of(2));
    assertNotEquals(Term.of(1.0), Term.of(2.0));
    assertNotEquals(Term.of(BigRational.ZERO), Term.of(BigRational.ONE));

    // Same values
    assertEquals(Term.of(9), Term.of(9));
    assertEquals(Term.of(9.0), Term.of(9.0));
    assertEquals(Term.of(BigRational.of(9)), Term.of(BigRational.of(9)));

    // Type conversion
    assertEquals(Term.of(9).toInteger().eval(map), Term.of(9));
    assertEquals(Term.of(9.0).toInteger().eval(map), Term.of(9));
    assertEquals(Term.of(BigRational.of(9)).toInteger().eval(map), Term.of(9));
    assertEquals(Term.of(9).toRational().eval(map), Term.of(BigRational.of(9)));
    assertEquals(Term.of(9.0).toRational().eval(map), Term.of(BigRational.of(9)));
    assertEquals(Term.of(BigRational.of(9)).toRational().eval(map), Term.of(BigRational.of(9)));
    assertEquals(Term.of(9).toReal().eval(map), Term.of(9.0));
    assertEquals(Term.of(9.0).toReal().eval(map), Term.of(9.0));
    assertEquals(Term.of(BigRational.of(9)).toReal().eval(map), Term.of(9.0));

    // Type check
    assertEquals(Term.of(Term.IS_INTEGER, Term.of(9)).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.IS_INTEGER, Term.of(9.0)).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.IS_INTEGER, Term.of(9.5)).eval(map), Term.FALSE);
    assertEquals(Term.of(Term.IS_INTEGER, Term.of(BigRational.of(9))).eval(map), Term.TRUE);
    assertEquals(Term.of(Term.IS_INTEGER, Term.of(BigRational.of(19, 2))).eval(map), Term.FALSE);

    // Other arithmetic
    assertEquals(Term.of(9).subtract(Term.of(8)).eval(map), Term.of(1));
    assertEquals(Term.of(9).multiply(Term.of(8)).eval(map), Term.of(72));
    assertEquals(Term.of(9).divide(Term.of(8)).eval(map), Term.of(BigRational.of(9, 8)));

    // divideTruncate
    assertEval(Term.of(5).divideTruncate(Term.of(3)), Term.of(1));
    assertEval(Term.of(-5).divideTruncate(Term.of(3)), Term.of(-1));
    assertEval(Term.of(5).divideTruncate(Term.of(-3)), Term.of(-1));
    assertEval(Term.of(-5).divideTruncate(Term.of(-3)), Term.of(1));

    // remainderTruncate
    assertEval(Term.of(5).remainderTruncate(Term.of(3)), Term.of(2));
    assertEval(Term.of(-5).remainderTruncate(Term.of(3)), Term.of(-2));
    assertEval(Term.of(5).remainderTruncate(Term.of(-3)), Term.of(2));
    assertEval(Term.of(-5).remainderTruncate(Term.of(-3)), Term.of(-2));

    // divideFloor
    assertEval(Term.of(5).divideFloor(Term.of(3)), Term.of(Math.floorDiv(5, 3)));
    assertEval(Term.of(-5).divideFloor(Term.of(3)), Term.of(Math.floorDiv(-5, 3)));
    assertEval(Term.of(5).divideFloor(Term.of(-3)), Term.of(Math.floorDiv(5, -3)));
    assertEval(Term.of(-5).divideFloor(Term.of(-3)), Term.of(Math.floorDiv(-5, -3)));
    assertEval(Term.of(5).divideFloor(Term.of(3)), Term.of(1));
    assertEval(Term.of(-5).divideFloor(Term.of(3)), Term.of(-2));
    assertEval(Term.of(5).divideFloor(Term.of(-3)), Term.of(-2));
    assertEval(Term.of(-5).divideFloor(Term.of(-3)), Term.of(1));

    // remainderFloor
    assertEval(Term.of(5).remainderFloor(Term.of(3)), Term.of(Math.floorMod(5, 3)));
    assertEval(Term.of(-5).remainderFloor(Term.of(3)), Term.of(Math.floorMod(-5, 3)));
    assertEval(Term.of(5).remainderFloor(Term.of(-3)), Term.of(Math.floorMod(5, -3)));
    assertEval(Term.of(-5).remainderFloor(Term.of(-3)), Term.of(Math.floorMod(-5, -3)));
    assertEval(Term.of(5).remainderFloor(Term.of(3)), Term.of(2));
    assertEval(Term.of(-5).remainderFloor(Term.of(3)), Term.of(1));
    assertEval(Term.of(5).remainderFloor(Term.of(-3)), Term.of(-1));
    assertEval(Term.of(-5).remainderFloor(Term.of(-3)), Term.of(-2));

    // divideEuclidean
    assertEval(Term.of(7).divideEuclidean(Term.of(3)), Term.of(2));
    assertEval(Term.of(7).divideEuclidean(Term.of(-3)), Term.of(-2));
    assertEval(Term.of(-7).divideEuclidean(Term.of(3)), Term.of(-3));
    assertEval(Term.of(-7).divideEuclidean(Term.of(-3)), Term.of(3));

    // remainderEuclidean
    assertEval(Term.of(7).remainderEuclidean(Term.of(3)), Term.of(1));
    assertEval(Term.of(7).remainderEuclidean(Term.of(-3)), Term.of(1));
    assertEval(Term.of(-7).remainderEuclidean(Term.of(3)), Term.of(2));
    assertEval(Term.of(-7).remainderEuclidean(Term.of(-3)), Term.of(2));

    // ceil
    assertEval(Term.of(BigRational.of("0")).ceil(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("1/10")).ceil(), Term.of(BigRational.of("1")));
    assertEval(Term.of(BigRational.of("5/10")).ceil(), Term.of(BigRational.of("1")));
    assertEval(Term.of(BigRational.of("9/10")).ceil(), Term.of(BigRational.of("1")));
    assertEval(Term.of(BigRational.of("-1/10")).ceil(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-5/10")).ceil(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-9/10")).ceil(), Term.of(BigRational.of("0")));

    // floor
    assertEval(Term.of(BigRational.of("0")).floor(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("1/10")).floor(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("5/10")).floor(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("9/10")).floor(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-1/10")).floor(), Term.of(BigRational.of("-1")));
    assertEval(Term.of(BigRational.of("-5/10")).floor(), Term.of(BigRational.of("-1")));
    assertEval(Term.of(BigRational.of("-9/10")).floor(), Term.of(BigRational.of("-1")));

    // round
    assertEval(Term.of(BigRational.of("0")).round(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("1/10")).round(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("5/10")).round(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("9/10")).round(), Term.of(BigRational.of("1")));
    assertEval(Term.of(BigRational.of("-1/10")).round(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-5/10")).round(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-9/10")).round(), Term.of(BigRational.of("-1")));

    // truncate
    assertEval(Term.of(BigRational.of("0")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("1/10")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("5/10")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("9/10")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-1/10")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-5/10")).truncate(), Term.of(BigRational.of("0")));
    assertEval(Term.of(BigRational.of("-9/10")).truncate(), Term.of(BigRational.of("0")));
  }

  @Test
  public void isAtom() {
    assertEquals(0, Term.of(0).size());
    assertEquals(0, Term.of(0.0).size());
    assertEquals(0, Term.of(BigRational.of(0)).size());
    assertEquals(0, Term.of(1).size());
    assertEquals(0, Term.of(1.0).size());
    assertEquals(0, Term.of(BigRational.of(1)).size());
    assertEquals(0, Term.ADD.size());
    assertEquals(0, new Function(Type.INTEGER, null).size());
    assertEquals(0, new Variable(Type.INTEGER).size());
    assertNotEquals(0, new Function(Type.INTEGER, null).add(new Variable(Type.INTEGER)).size());
  }

  @Test
  public void isConstant() {
    assertTrue(Term.of(0).isConstant());
    assertTrue(Term.of(0.0).isConstant());
    assertTrue(Term.of(BigRational.of(0)).isConstant());
    assertTrue(Term.of(1).isConstant());
    assertTrue(Term.of(1.0).isConstant());
    assertTrue(Term.of(BigRational.of(1)).isConstant());
    assertTrue(Term.ADD.isConstant());
    assertFalse(new Function(Type.INTEGER, null).isConstant());
    assertFalse(new Variable(Type.INTEGER).isConstant());
    assertFalse(new Function(Type.INTEGER, null).add(new Variable(Type.INTEGER)).isConstant());
  }

  @Test
  public void isOne() {
    assertFalse(Term.of(0).isOne());
    assertFalse(Term.of(0.0).isOne());
    assertFalse(Term.of(BigRational.of(0)).isOne());
    assertTrue(Term.of(1).isOne());
    assertTrue(Term.of(1.0).isOne());
    assertTrue(Term.of(BigRational.of(1)).isOne());
    assertFalse(Term.ADD.isOne());
    assertFalse(new Function(Type.INTEGER, null).isOne());
    assertFalse(new Variable(Type.INTEGER).isOne());
    assertFalse(new Function(Type.INTEGER, null).add(new Variable(Type.INTEGER)).isOne());
  }

  @Test
  public void isZero() {
    assertTrue(Term.of(0).isZero());
    assertTrue(Term.of(0.0).isZero());
    assertTrue(Term.of(BigRational.of(0)).isZero());
    assertFalse(Term.of(1).isZero());
    assertFalse(Term.of(1.0).isZero());
    assertFalse(Term.of(BigRational.of(1)).isZero());
    assertFalse(new Variable(Type.INTEGER).isZero());
    assertFalse(Term.ADD.isZero());
    assertFalse(new Function(Type.INTEGER, null).isZero());
    assertFalse(new Variable(Type.INTEGER).isZero());
    assertFalse(new Function(Type.INTEGER, null).add(new Variable(Type.INTEGER)).isZero());
  }

  @Test
  public void isomorphic() {
    var a = new Function(Type.INDIVIDUAL, "a");
    var b = new Function(Type.INDIVIDUAL, "b");
    var f = new Function(Type.of(Type.BOOLEAN, Type.INDIVIDUAL), "f");
    var r = new Variable(Type.REAL);
    var x = new Variable(Type.INDIVIDUAL);
    var y = new Variable(Type.INDIVIDUAL);
    Map<Variable, Variable> map;

    // Atoms, equal
    map = new HashMap<>();
    assertTrue(a.isomorphic(a, map));
    assertEquals(map.size(), 0);

    // Atoms, unequal
    map = new HashMap<>();
    assertFalse(a.isomorphic(b, map));

    // Variables, equal
    map = new HashMap<>();
    assertTrue(x.isomorphic(x, map));
    assertEquals(map.size(), 0);

    // Variables, match
    map = new HashMap<>();
    assertTrue(x.isomorphic(y, map));
    assertEquals(map.size(), 2);

    // Variables, different types
    map = new HashMap<>();
    assertFalse(x.isomorphic(r, map));

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Term.of(Term.EQ, a, a).isomorphic(Term.of(Term.EQ, a, a), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Term.of(Term.EQ, x, x).isomorphic(Term.of(Term.EQ, x, x), map));
    assertEquals(map.size(), 0);

    // Compound, equal
    map = new HashMap<>();
    assertTrue(Term.of(Term.EQ, a, f.call(x)).isomorphic(Term.of(Term.EQ, a, f.call(x)), map));
    assertEquals(map.size(), 0);

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Term.of(Term.EQ, a, a).isomorphic(Term.of(Term.EQ, a, b), map));

    // Compound, unequal
    map = new HashMap<>();
    assertFalse(Term.of(Term.EQ, a, a).isomorphic(Term.of(Term.EQ, a, x), map));

    // Compound, match
    map = new HashMap<>();
    assertTrue(Term.of(Term.EQ, a, f.call(x)).isomorphic(Term.of(Term.EQ, a, f.call(y)), map));
    assertEquals(map.size(), 2);
  }

  @Test
  public void match() {

    // Subset of unify (see below)
    // Gives different results in several cases
    // In particular, has no notion of an occurs check
    // Assumes the inputs have disjoint variables
    var a = new Function(Type.INDIVIDUAL, "a");
    var b = new Function(Type.INDIVIDUAL, "b");
    var f1 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL), "f1");
    var f2 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL, Type.INDIVIDUAL), "f2");
    var g1 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL), "g1");
    var x = new Variable(Type.INDIVIDUAL);
    var y = new Variable(Type.INDIVIDUAL);
    var z = new Variable(Type.INDIVIDUAL);
    Map<Variable, Term> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(a.match(a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(a.match(b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(x.match(x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertFalse(a.match(x, map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(x.match(y, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), y.replaceVars(map));

    // function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(f2.call(a, x).match(f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), b);

    // f1 and g1 do not match
    map = new HashMap<>();
    assertFalse(f1.call(a).match(g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(f1.call(x).match(f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), y.replaceVars(map));

    // f1 and g1 do not match
    map = new HashMap<>();
    assertFalse(f1.call(x).match(g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(f1.call(x).unify(f2.call(y, z), map));

    // Unifies y with the term g1(x)
    map = new HashMap<>();
    assertFalse(f1.call(g1.call(x)).match(f1.call(y), map));

    // Unifies x with constant a, and y with the term g1(a)
    map = new HashMap<>();
    assertFalse(f2.call(g1.call(x), x).match(f2.call(y, a), map));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertTrue(x.match(f1.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(x.match(y, map));
    assertTrue(y.match(a, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replaceVars(map), a);
    assertEquals(y.replaceVars(map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertFalse(a.match(y, map));

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(x.match(a, map));
    assertFalse(b.match(x, map));
  }

  @Test
  public void simplify() {

    // Constant expressions
    assertEquals(Term.of(1).add(Term.of(2)).simplify(), Term.of(3));
    assertEquals(Term.of(1).add(Term.of(2)).add(Term.of(3)).simplify(), Term.of(6));
    assertEquals(Term.of(1.0).add(Term.of(2.0)).add(Term.of(3.0)).simplify(), Term.of(6.0));
    var third = Term.of(BigRational.of(1, 3));
    var one = Term.of(BigRational.ONE);
    assertEquals(third.add(third).add(third).simplify(), one);

    // Arithmetic identities
    var y = new Variable(Type.INTEGER);
    assertEquals(Term.of(Term.ADD, Term.of(0), y).simplify(), y);
    assertEquals(Term.of(Term.ADD, y, Term.of(0)).simplify(), y);
    assertEquals(Term.of(Term.SUBTRACT, Term.of(0), y).simplify(), Term.of(Term.NEGATE, y));
    assertEquals(Term.of(Term.SUBTRACT, y, Term.of(0)).simplify(), y);

    // Equality
    assertEquals(y.eq(y).simplify(), Term.TRUE);
    assertEquals(y.notEq(y).simplify(), Term.FALSE);
  }

  @Test
  public void splice() {
    var position = new ArrayList<Integer>();

    // 1
    assertEquals(Term.of(1).splice(position, Term.of(9)), Term.of(9));

    // (+ 1 2)
    position.add(1);
    assertEquals(
        Term.of(1).add(Term.of(2)).splice(position, Term.of(9)), Term.of(9).add(Term.of(2)));
  }

  @Test
  public void type() {
    assertEquals(Term.of(1.0).type(), Type.REAL);
    assertEquals(Term.of(Term.NOT, Term.TRUE).type(), Type.BOOLEAN);
    assertEquals(Term.of(Term.AND, Term.TRUE, Term.FALSE).type(), Type.BOOLEAN);
    assertEquals(Term.of(Term.NEGATE, Term.of(1.0)).type(), Type.REAL);
    assertEquals(Term.of(Term.ADD, Term.of(1.0), Term.of(1.0)).type(), Type.REAL);
    assertEquals(Term.of(Term.EQ, Term.of(1.0), Term.of(1.0)).type(), Type.BOOLEAN);
    assertEquals(Term.of(Term.LESS, Term.of(1.0), Term.of(1.0)).type(), Type.BOOLEAN);
    var x = new Variable(Type.REAL);
    assertEquals(x.type(), Type.REAL);
  }

  @Test
  public void unequal() {
    var a = new DistinctObject("a");
    var b = new DistinctObject("b");
    var f = new Function(Type.INDIVIDUAL, "f");
    var g = new Function(Type.INDIVIDUAL, "g");
    var x = new Variable(Type.INDIVIDUAL);
    var y = new Variable(Type.INDIVIDUAL);

    // Unequal
    assertTrue(a.unequal(b));
    assertTrue(Term.of(BigInteger.ONE).unequal(Term.of(BigInteger.TWO)));
    assertTrue(Term.of(BigRational.of("1/2")).unequal(Term.of(BigRational.of("1/3"))));

    // Equal
    assertFalse(a.unequal(a));
    assertFalse(Term.of(BigInteger.ONE).unequal(Term.of(BigInteger.ONE)));
    assertFalse(Term.of(BigRational.of("1/2")).unequal(Term.of(BigRational.of("1/2"))));
    assertFalse(f.unequal(f));
    assertFalse(x.unequal(x));

    // Unknown
    assertFalse(f.unequal(g));
    assertFalse(f.unequal(x));
    assertFalse(x.unequal(y));
  }

  @Test
  public void unify() {

    // https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
    var a = new Function(Type.INDIVIDUAL, "a");
    var b = new Function(Type.INDIVIDUAL, "b");
    var f1 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL), "f1");
    var f2 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL, Type.INDIVIDUAL), "f2");
    var g1 = new Function(Type.of(Type.INDIVIDUAL, Type.INDIVIDUAL), "g1");
    var x = new Variable(Type.INDIVIDUAL);
    var y = new Variable(Type.INDIVIDUAL);
    var z = new Variable(Type.INDIVIDUAL);
    Map<Variable, Term> map;

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(a.unify(a, map));
    assertEquals(map.size(), 0);

    // a and b do not match
    map = new HashMap<>();
    assertFalse(a.unify(b, map));

    // Succeeds. (tautology)
    map = new HashMap<>();
    assertTrue(x.unify(x, map));
    assertEquals(map.size(), 0);

    // x is unified with the constant a
    map = new HashMap<>();
    assertTrue(a.unify(x, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), a);

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(x.unify(y, map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), y.replaceVars(map));

    // function and constant symbols match, x is unified with the constant b
    map = new HashMap<>();
    assertTrue(f2.call(a, x).unify(f2.call(a, b), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), b);

    // f1 and g1 do not match
    map = new HashMap<>();
    assertFalse(f1.call(a).unify(g1.call(a), map));

    // x and y are aliased
    map = new HashMap<>();
    assertTrue(f1.call(x).unify(f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(x.replaceVars(map), y.replaceVars(map));

    // f1 and g1 do not match
    map = new HashMap<>();
    assertFalse(f1.call(x).unify(g1.call(y), map));

    // Fails. The f function symbols have different arity
    map = new HashMap<>();
    assertFalse(f1.call(x).unify(f2.call(y, z), map));

    // Unifies y with the term g1(x)
    map = new HashMap<>();
    assertTrue(f1.call(g1.call(x)).unify(f1.call(y), map));
    assertEquals(map.size(), 1);
    assertEquals(y.replaceVars(map), g1.call(x));

    // Unifies x with constant a, and y with the term g1(a)
    map = new HashMap<>();
    assertTrue(f2.call(g1.call(x), x).unify(f2.call(y, a), map));
    assertEquals(map.size(), 2);
    assertEquals(x.replaceVars(map), a);
    assertEquals(y.replaceVars(map), g1.call(a));

    // Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs
    // check).
    map = new HashMap<>();
    assertFalse(x.unify(f1.call(x), map));

    // Both x and y are unified with the constant a
    map = new HashMap<>();
    assertTrue(x.unify(y, map));
    assertTrue(y.unify(a, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replaceVars(map), a);
    assertEquals(y.replaceVars(map), a);

    // As above (order of equations in set doesn't matter)
    map = new HashMap<>();
    assertTrue(a.unify(y, map));
    assertTrue(x.unify(y, map));
    assertEquals(map.size(), 2);
    assertEquals(x.replaceVars(map), a);
    assertEquals(y.replaceVars(map), a);

    // Fails. a and b do not match, so x can't be unified with both
    map = new HashMap<>();
    assertTrue(x.unify(a, map));
    assertFalse(b.unify(x, map));
  }
}
