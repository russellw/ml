package prover;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;

public final class TptpPrinter {
  private static void print(Symbol a) {
    switch (a) {
      case BOOLEAN:
        System.out.print("$o");
        return;
      case INDIVIDUAL:
        System.out.print("$i");
        return;
      case ADD:
        System.out.print("$sum");
        return;
      case CEIL:
        System.out.print("$ceiling");
        return;
      case DIVIDE:
        System.out.print("$quotient");
        return;
      case DIVIDE_EUCLIDEAN:
        System.out.print("$quotient_e");
        return;
      case DIVIDE_FLOOR:
        System.out.print("$quotient_f");
        return;
      case DIVIDE_TRUNCATE:
        System.out.print("$quotient_t");
        return;
      case FLOOR:
        System.out.print("$floor");
        return;
      case IS_INTEGER:
        System.out.print("$is_int");
        return;
      case INTEGER:
        System.out.print("$int");
        return;
      case IS_RATIONAL:
        System.out.print("$is_rat");
        return;
      case RATIONAL:
        System.out.print("$rat");
        return;
      case LESS:
        System.out.print("$lesseq");
        return;
      case LESS_EQ:
        System.out.print("$less");
        return;
      case MULTIPLY:
        System.out.print("$product");
        return;
      case NEGATE:
        System.out.print("$negate");
        return;
      case REMAINDER_EUCLIDEAN:
        System.out.print("$remainder_e");
        return;
      case REMAINDER_FLOOR:
        System.out.print("$remainder_f");
        return;
      case REMAINDER_TRUNCATE:
        System.out.print("$remainder_t");
        return;
      case ROUND:
        System.out.print("$round");
        return;
      case SUBTRACT:
        System.out.print("$difference");
        return;
      case TO_INTEGER:
        System.out.print("$to_int");
        return;
      case TO_RATIONAL:
        System.out.print("$to_rat");
        return;
      case TO_REAL:
        System.out.print("$to_real");
        return;
      case REAL:
        System.out.print("$real");
        return;
      case TRUNCATE:
        System.out.print("$truncate");
        return;
    }
  }

  private static boolean needParens(List<Object> a, List<Object> parent) {
    if (parent == null) return false;
    var op = a.get(0);
    if (op instanceof Symbol)
      switch ((Symbol) op) {
        case AND:
        case EQV:
        case OR:
          {
            var parentOp = parent.get(0);
            if (parentOp instanceof Symbol)
              switch ((Symbol) parentOp) {
                case ALL:
                case AND:
                case EQV:
                case EXISTS:
                case NOT:
                case OR:
                  return true;
              }
            break;
          }
      }
    return false;
  }

  private static void infix(String op, List<Object> a) {
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) System.out.print(op);
      print(a.get(i), a);
    }
  }

  private static void quant(List<Object> a) {
    System.out.print('[');
    var binding = (List) a.get(1);
    for (var i = 0; i < binding.size(); i++) {
      var x = binding.get(i);
      if (i > 0) System.out.print(',');
      print(x);
      var type = Types.typeof(x);
      if (type != Symbol.INDIVIDUAL) {
        System.out.print(':');
        print(type);
      }
    }
    System.out.print("]:");
    print(a.get(2), a);
  }

  private static void print(Object a) {
    print(a, null);
  }

  public static void proof(Clause refutation) {
    var proof = refutation.proof();

    // Names for anonymous formulas
    var i = -1L;
    for (var formula : proof)
      if (formula.name instanceof Long) i = Math.max(i, (long) formula.name);
    for (var formula : proof) if (formula.name == null) formula.name = ++i;

    // Print
    for (var formula : proof) println(formula);
  }

  public static void println(AbstractFormula formula) {
    System.out.print(formula instanceof Clause ? "cnf(" : "fof(");
    System.out.print(formula.name);
    System.out.print(", ");

    // Role
    switch (formula.inference) {
      case CONJECTURE:
        System.out.print("conjecture");
        break;
      case NEGATE:
        System.out.print("negated_conjecture");
        break;
      default:
        System.out.print("plain");
        break;
    }
    System.out.print(", ");

    // Term
    Variable.names.clear();
    print(formula.term());
    System.out.print(", ");

    // Source
    switch (formula.inference) {
      case RENAME_VARIABLES:
        throw new IllegalArgumentException(formula.toString());
      case DEFINE:
        System.out.print("introduced(definition)");
        break;
      case CONJECTURE:
      case AXIOM:
        System.out.printf(
            "file(%s,%s)",
            Etc.quote('\'', Path.of(formula.file).getFileName().toString()), formula.name);
        break;
      case NEGATE:
        System.out.printf("inference(negate,[status(ceq)],[%s])", formula.from[0].name);
        break;
      default:
        System.out.printf(
            "inference(%s,[status(", formula.inference.toString().toLowerCase(Locale.ROOT));

        // If a formula introduces new symbols, then it is only equisatisfiable
        // This happens during subformula renaming in CNF conversion
        var fromFuncs = new HashSet<>();
        for (var from : formula.from)
          Etc.collectLeaves(from.term(), a -> a instanceof Func, fromFuncs);
        var funcs = Etc.collectLeaves(formula.term(), a -> a instanceof Func);
        System.out.print(fromFuncs.containsAll(funcs) ? "thm" : "esa");

        System.out.print(")],[");
        for (var i = 0; i < formula.from.length; i++) {
          if (i > 0) System.out.print(',');
          System.out.print(formula.from[i].name);
        }
        System.out.print("])");
        break;
    }
    System.out.println(").");
  }

  private static boolean isWeird(String s) {
    if (Character.isDigit(s.charAt(0))) return true;
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if (!(Character.isLetterOrDigit(c) || c == '_')) return true;
    }
    return false;
  }

  private static void args(List<Object> a) {
    System.out.print('(');
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) System.out.print(',');
      print(a.get(i));
    }
    System.out.print(')');
  }

  @SuppressWarnings("unchecked")
  private static void print(Object a, List<Object> parent) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol) {
        if (needParens(a1, parent)) System.out.print('(');
        switch ((Symbol) op) {
          case ALL:
            System.out.print('!');
            quant(a1);
            break;
          case EXISTS:
            System.out.print('?');
            quant(a1);
            break;
          case AND:
            infix(" & ", a1);
            break;
          case OR:
            infix(" | ", a1);
            break;
          case NOT:
            if (Etc.head(a1.get(1)) == Symbol.EQUALS) {
              infix("!=", (List) a1.get(1));
              break;
            }
            System.out.print('~');
            print(a1.get(1), a1);
            break;
          case EQUALS:
            infix("=", a1);
            break;
          case EQV:
            infix(" <=> ", a1);
            break;
          default:
            print(op);
            args(a1);
            return;
        }
        if (needParens(a1, parent)) System.out.print(')');
        return;
      }
      print(op);
      args(a1);
      return;
    }
    if (a instanceof Func) {
      var name = a.toString();
      if (isWeird(name)) {
        System.out.print(Etc.quote('\'', name));
        return;
      }
    }
    if (a instanceof Symbol) {
      print((Symbol) a);
      return;
    }
    if (a instanceof Boolean) System.out.print('$');
    if (a instanceof String) {
      System.out.print(Etc.quote('"', (String) a));
      return;
    }
    System.out.print(a);
  }
}
