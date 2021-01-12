package prover;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public final class TptpPrinter {
  private final PrintStream out = System.out;

  private void print(Symbol a) {
    switch (a) {
      case BOOLEAN:
        out.print("$o");
        return;
      case INDIVIDUAL:
        out.print("$i");
        return;
      case ADD:
        out.print("$sum");
        return;
      case CEIL:
        out.print("$ceiling");
        return;
      case DIVIDE:
        out.print("$quotient");
        return;
      case DIVIDE_EUCLIDEAN:
        out.print("$quotient_e");
        return;
      case DIVIDE_FLOOR:
        out.print("$quotient_f");
        return;
      case DIVIDE_TRUNCATE:
        out.print("$quotient_t");
        return;
      case FLOOR:
        out.print("$floor");
        return;
      case IS_INTEGER:
        out.print("$is_int");
        return;
      case INTEGER:
        out.print("$int");
        return;
      case IS_RATIONAL:
        out.print("$is_rat");
        return;
      case RATIONAL:
        out.print("$rat");
        return;
      case LESS:
        out.print("$lesseq");
        return;
      case LESS_EQ:
        out.print("$less");
        return;
      case MULTIPLY:
        out.print("$product");
        return;
      case NEGATE:
        out.print("$negate");
        return;
      case REMAINDER_EUCLIDEAN:
        out.print("$remainder_e");
        return;
      case REMAINDER_FLOOR:
        out.print("$remainder_f");
        return;
      case REMAINDER_TRUNCATE:
        out.print("$remainder_t");
        return;
      case ROUND:
        out.print("$round");
        return;
      case SUBTRACT:
        out.print("$difference");
        return;
      case TO_INTEGER:
        out.print("$to_int");
        return;
      case TO_RATIONAL:
        out.print("$to_rat");
        return;
      case TO_REAL:
        out.print("$to_real");
        return;
      case REAL:
        out.print("$real");
        return;
      case TRUNCATE:
        out.print("$truncate");
        return;
    }
  }

  private boolean needParens(List<Object> a, List<Object> parent) {
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

  private void infix(String op, List<Object> a) {
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) out.print(op);
      print(a.get(i), a);
    }
  }

  private void quant(List<Object> a) {
    out.print('[');
    var binding = (List) a.get(1);
    for (var i = 0; i < binding.size(); i++) {
      var x = binding.get(i);
      if (i > 0) out.print(',');
      print(x);
      var type = Types.typeof(x);
      if (type != Symbol.INDIVIDUAL) {
        out.print(':');
        print(type);
      }
    }
    out.print("]:");
    print(a.get(2), a);
  }

  private void print(Object a) {
    print(a, null);
  }

  public void proof(String file, Clause refutation) {
    out.println("% SZS output start CNFRefutation for " + file);
    var proof = refutation.proof();

    // Names for anonymous formulas
    var i = -1L;
    for (var formula : proof)
      if (formula.name instanceof Long) i = Math.max(i, (long) formula.name);
    for (var formula : proof) if (formula.name == null) formula.name = ++i;

    // Names for anonymous Skolem functions
    var pattern = Pattern.compile("sK\\d+");
    var skolems = new LinkedHashSet<Func>();
    var j = new long[] {-1L};
    for (var formula : proof)
      Etc.walk(
          formula.term(),
          a -> {
            if (a instanceof Func) {
              var a1 = (Func) a;
              var name = a1.name;
              if (name == null) {
                skolems.add(a1);
                return;
              }
              var matcher = pattern.matcher(name);
              if (matcher.matches()) j[0] = Math.max(j[0], Long.parseLong(matcher.group(1)));
            }
          });
    i = j[0];
    for (var a : skolems) a.name = "sK" + ++i;

    // Print
    for (var formula : proof) println(formula);
    out.println("% SZS output end CNFRefutation for " + file);
  }

  public void println(AbstractFormula formula) {
    out.print(formula instanceof Clause ? "cnf(" : "fof(");
    out.print(formula.name);
    out.print(", ");

    // Role
    switch (formula.inference) {
      case CONJECTURE:
        out.print("conjecture");
        break;
      case NEGATE:
        out.print("negated_conjecture");
        break;
      default:
        out.print("plain");
        break;
    }
    out.print(", ");

    // Term
    Variable.names.clear();
    print(formula.term());
    out.print(", ");

    // Source
    switch (formula.inference) {
      case RENAME_VARIABLES:
        throw new IllegalArgumentException(formula.toString());
      case DEFINE:
        out.print("introduced(definition)");
        break;
      case CONJECTURE:
      case AXIOM:
        out.printf(
            "file(%s,%s)",
            Etc.quote('\'', Path.of(formula.file).getFileName().toString()), formula.name);
        break;
      case NEGATE:
        out.printf("inference(negate,[status(ceq)],[%s])", formula.from[0].name);
        break;
      default:
        out.printf("inference(%s,[status(", formula.inference.toString().toLowerCase(Locale.ROOT));

        // If a formula introduces new symbols, then it is only equisatisfiable
        // This happens during subformula renaming in CNF conversion
        var fromFuncs = new HashSet<>();
        for (var from : formula.from) Etc.collect(from.term(), a -> a instanceof Func, fromFuncs);
        var funcs = Etc.collect(formula.term(), a -> a instanceof Func);
        out.print(fromFuncs.containsAll(funcs) ? "thm" : "esa");

        out.print(")],[");
        for (var i = 0; i < formula.from.length; i++) {
          if (i > 0) out.print(',');
          out.print(formula.from[i].name);
        }
        out.print("])");
        break;
    }
    out.println(").");
  }

  private boolean isWeird(String s) {
    if (Character.isDigit(s.charAt(0))) return true;
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if (!(Character.isLetterOrDigit(c) || c == '_')) return true;
    }
    return false;
  }

  private void args(List<Object> a) {
    out.print('(');
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) out.print(',');
      print(a.get(i));
    }
    out.print(')');
  }

  @SuppressWarnings("unchecked")
  private void print(Object a, List<Object> parent) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol) {
        if (needParens(a1, parent)) out.print('(');
        switch ((Symbol) op) {
          case ALL:
            out.print('!');
            quant(a1);
            break;
          case EXISTS:
            out.print('?');
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
            out.print('~');
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
        if (needParens(a1, parent)) out.print(')');
        return;
      }
      print(op);
      args(a1);
      return;
    }
    if (a instanceof Func) {
      var name = a.toString();
      if (isWeird(name)) {
        out.print(Etc.quote('\'', name));
        return;
      }
    }
    if (a instanceof Symbol) {
      print((Symbol) a);
      return;
    }
    if (a instanceof Boolean) out.print('$');
    if (a instanceof String) {
      out.print(Etc.quote('"', (String) a));
      return;
    }
    out.print(a);
  }
}
