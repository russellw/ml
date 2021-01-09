package prover;

import io.vavr.collection.HashSet;
import io.vavr.collection.Seq;
import io.vavr.collection.Set;
import java.util.HashMap;

public final class Variable {
  public final Object type;
  public static final HashMap<Variable, String> names = new HashMap<>();

  public Variable(Object type) {
    this.type = type;
  }

  @SuppressWarnings("unchecked")
  private static void getFreeVariables(
      Set<Variable> bound, Object a, java.util.HashSet<Variable> r) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
      if (op instanceof Symbol)
        switch ((Symbol) op) {
          case ALL:
          case EXISTS:
            {
              var binding = (Seq) a1.get(1);
              getFreeVariables(bound.addAll(binding), a1.get(2), r);
              return;
            }
        }
      for (var b : a1) getFreeVariables(bound, b, r);
      return;
    }
    if (a instanceof Variable) {
      var a1 = (Variable) a;
      if (!bound.contains(a1)) r.add(a1);
    }
  }

  public static java.util.HashSet<Variable> freeVariables(Object a) {
    var r = new java.util.HashSet<Variable>();
    getFreeVariables(HashSet.empty(), a, r);
    return r;
  }

  public static Object unquantify(Object a) {
    while (a instanceof Seq) {
      var a1 = (Seq) a;
      if (a1.head() != Symbol.ALL) break;
      a = a1.get(2);
    }
    return a;
  }

  @Override
  public String toString() {
    var name = names.get(this);
    if (name == null) {
      var i = names.size();
      name = i < 26 ? Character.toString('A' + i) : "Z" + (i - 25);
      names.put(this, name);
    }
    return name;
  }
}
