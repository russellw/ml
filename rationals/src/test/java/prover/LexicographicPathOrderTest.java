package prover;

import static org.junit.Assert.*;

import java.util.ArrayList;
import org.junit.Test;

public class LexicographicPathOrderTest {
  @Test
  public void greater() {
    Term a, b;

    // Multiplication and addition
    var s = new Function(Type.of(Type.INTEGER, Type.INTEGER), "s");
    var add = new Function(Type.of(Type.INTEGER, Type.INTEGER, Type.INTEGER), "add");
    var mul = new Function(Type.of(Type.INTEGER, Type.INTEGER, Type.INTEGER), "mul");
    var ops = new ArrayList<Term>();
    ops.add(s);
    ops.add(add);
    ops.add(mul);
    var lpo = new LexicographicPathOrder(ops);
    var x = new Variable(Type.INTEGER);
    var y = new Variable(Type.INTEGER);

    // 0 + y = y
    a = add.call(Term.of(0), y);
    b = y;
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // s(x) + y = s(x + y)
    a = add.call(s.call(x), y);
    b = s.call(add.call(x, y));
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // 0 * y = 0
    a = mul.call(Term.of(0), y);
    b = Term.of(0);
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // s(x) * y = (x * y) + y
    a = mul.call(s.call(x), y);
    b = add.call(mul.call(x, y), y);
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // Ackermann
    var ack = new Function(Type.of(Type.INTEGER, Type.INTEGER, Type.INTEGER), "ack");
    ops = new ArrayList<>();
    ops.add(s);
    ops.add(ack);
    lpo = new LexicographicPathOrder(ops);

    // ack(0,0) = 0
    a = ack.call(Term.of(0), Term.of(0));
    b = Term.of(0);
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // ack(0,s(y)) = s(s(ack(0,y)))
    a = ack.call(Term.of(0), s.call(y));
    b = s.call(s.call(ack.call(Term.of(0), y)));
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // ack(s(x),0) = s(0)
    a = ack.call(s.call(x), Term.of(0));
    b = s.call(Term.of(0));
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // ack(s(x),s(y)) = ack(x,ack(s(x),y))
    a = ack.call(s.call(x), s.call(y));
    b = ack.call(x, ack.call(s.call(x), y));
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // Inverse
    var e = new Function(Type.REAL, "e");
    mul = new Function(Type.of(Type.REAL, Type.REAL, Type.REAL), "mul");
    var inv = new Function(Type.of(Type.REAL, Type.REAL), "inv");
    ops = new ArrayList<>();
    ops.add(e);
    ops.add(mul);
    ops.add(inv);
    lpo = new LexicographicPathOrder(ops);
    x = new Variable(Type.REAL);
    y = new Variable(Type.REAL);
    var z = new Variable(Type.REAL);

    // e * x = x
    a = mul.call(e, x);
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * e = x
    a = mul.call(x, e);
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x) * x = e
    a = mul.call(inv.call(x), x);
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * inv(x) = e
    a = mul.call(x, inv.call(x));
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // (x * y) * z = x * (y * z)
    a = mul.call(mul.call(x, y), z);
    b = mul.call(x, mul.call(y, z));
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(inv(x)) = x
    a = inv.call(inv.call(x));
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(e) = e
    a = inv.call(e);
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x * y) = inv(y) * inv(x)
    a = inv.call(mul.call(x, y));
    b = mul.call(inv.call(y), inv.call(x));
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x) * (x * y) = y
    a = mul.call(inv.call(x), mul.call(x, y));
    b = y;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * (inv(x) * y) = y
    a = mul.call(x, mul.call(inv.call(x), y));
    b = y;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // Syntactical
    var f = new Function(Type.of(Type.REAL, Type.REAL), "f");
    var g = new Function(Type.REAL, "g");
    ops = new ArrayList<>();
    ops.add(f);
    ops.add(g);
    lpo = new LexicographicPathOrder(ops);

    // f(x) = g
    a = f.call(x);
    b = g;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertFalse(lpo.greater(a, b));
  }

  @Test
  public void greaterBuiltin() {
    Term a, b;

    // Now using the built-in multiplication and addition ops
    var s = new Function(Type.of(Type.INTEGER, Type.INTEGER), "s");
    var add = Term.ADD;
    var mul = Term.MULTIPLY;
    var ops = new ArrayList<Term>();
    ops.add(s);
    ops.add(add);
    ops.add(mul);
    var lpo = new LexicographicPathOrder(ops);
    var x = new Variable(Type.INTEGER);
    var y = new Variable(Type.INTEGER);

    // 0 + y = y
    a = Term.of(0).add(y);
    b = y;
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // s(x) + y = s(x + y)
    a = s.call(x).add(y);
    b = s.call(x.add(y));
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // 0 * y = 0
    a = Term.of(0).multiply(y);
    b = Term.of(0);
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // s(x) * y = (x * y) + y
    a = s.call(x).multiply(y);
    b = x.multiply(y).add(y);
    a.typeCheck(Type.INTEGER);
    b.typeCheck(Type.INTEGER);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // Inverse
    var e = new Function(Type.REAL, "e");
    var inv = new Function(Type.of(Type.REAL, Type.REAL), "inv");
    ops = new ArrayList<>();
    ops.add(e);
    ops.add(mul);
    ops.add(inv);
    lpo = new LexicographicPathOrder(ops);
    x = new Variable(Type.REAL);
    y = new Variable(Type.REAL);
    var z = new Variable(Type.REAL);

    // e * x = x
    a = e.multiply(x);
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * e = x
    a = x.multiply(e);
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x) * x = e
    a = inv.call(x).multiply(x);
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * inv(x) = e
    a = x.multiply(inv.call(x));
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // (x * y) * z = x * (y * z)
    a = x.multiply(y).multiply(z);
    b = x.multiply(y.multiply(z));
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(inv(x)) = x
    a = inv.call(inv.call(x));
    b = x;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(e) = e
    a = inv.call(e);
    b = e;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x * y) = inv(y) * inv(x)
    a = inv.call(x.multiply(y));
    b = inv.call(y).multiply(inv.call(x));
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // inv(x) * (x * y) = y
    a = inv.call(x).multiply(x.multiply(y));
    b = y;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));

    // x * (inv(x) * y) = y
    a = x.multiply(inv.call(x).multiply(y));
    b = y;
    a.typeCheck(Type.REAL);
    b.typeCheck(Type.REAL);
    assertFalse(lpo.greater(a, a));
    assertFalse(lpo.greater(b, a));
    assertFalse(lpo.greater(b, b));
    assertTrue(lpo.greater(a, b));
  }
}
