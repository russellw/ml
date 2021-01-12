package prover;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public final class TptpPrinter {
  private final PrintStream stream = System.out;

  private void print(Symbol a) {
    switch (a) {
      case BOOLEAN:
        stream.print("$o");
        return;
      case INDIVIDUAL:
        stream.print("$i");
        return;
      case ADD:
        stream.print("$sum");
        return;
      case ALL:
        stream.print("!");
        return;
      case AND:
        stream.print("&");
        return;
      case CEIL:
        stream.print("$ceiling");
        return;
      case DIVIDE:
        stream.print("$quotient");
        return;
      case DIVIDE_EUCLIDEAN:
        stream.print("$quotient_e");
        return;
      case DIVIDE_FLOOR:
        stream.print("$quotient_f");
        return;
      case DIVIDE_TRUNCATE:
        stream.print("$quotient_t");
        return;
      case EQUALS:
        stream.print("=");
        return;
      case EQV:
        stream.print("<=>");
        return;
      case EXISTS:
        stream.print("?");
        return;
      case FLOOR:
        stream.print("$floor");
        return;
      case IS_INTEGER:
        stream.print("$is_int");
        return;
      case INTEGER:
        stream.print("$int");
        return;
      case IS_RATIONAL:
        stream.print("$is_rat");
        return;
      case RATIONAL:
        stream.print("rat");
        return;
      case LESS:
        stream.print("$lesseq");
        return;
      case LESS_EQ:
        stream.print("$less");
        return;
      case MULTIPLY:
        stream.print("$product");
        return;
      case NEGATE:
        stream.print("$negate");
        return;
      case NOT:
        stream.print("~");
        return;
      case OR:
        stream.print("|");
        return;
      case REMAINDER_EUCLIDEAN:
        stream.print("$remainder_e");
        return;
      case REMAINDER_FLOOR:
        stream.print("$remainder_f");
        return;
      case REMAINDER_TRUNCATE:
        stream.print("$remainder_t");
        return;
      case ROUND:
        stream.print("$round");
        return;
      case SUBTRACT:
        stream.print("$difference");
        return;
      case TO_INTEGER:
        stream.print("$to_int");
        return;
      case TO_RATIONAL:
        stream.print("$to_rat");
        return;
      case TO_REAL:
        stream.print("$to_real");
        return;
      case REAL:
        stream.print("$real");
        return;
      case TRUNCATE:
        stream.print("$truncate");
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
      if (i > 1) stream.print(op);
      print(a.get(i), a);
    }
  }

  private void quant(List<Object> a) {
    stream.print('[');
    var binding = (List) a.get(1);
    for (var i = 0; i < binding.size(); i++) {
      var x = binding.get(i);
      if (i > 0) stream.print(',');
      print(x);
      var type = Types.typeof(x);
      if (type != Symbol.INDIVIDUAL) {
        stream.print(':');
        print(type);
      }
    }
    stream.print("]:");
    print(a.get(2), a);
  }

  private void print(Object a) {
    print(a, null);
  }

  public void proof(String file, Clause refutation) {
    stream.println("% SZS output start CNFRefutation for " + file);
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
    stream.println("% SZS output end CNFRefutation for " + file);
  }

  public void println(AbstractFormula formula) {
    stream.print(formula instanceof Clause ? "cnf(" : "fof(");
    stream.print(formula.name);
    stream.print(", ");

    // Role
    switch (formula.inference) {
      case CONJECTURE:
        stream.print("conjecture");
        break;
      case NEGATE:
        stream.print("negated_conjecture");
        break;
      default:
        stream.print("plain");
        break;
    }
    stream.print(", ");

    // Term
    Variable.names.clear();
    print(formula.term());
    stream.print(", ");

    // Source
    switch (formula.inference) {
      case RENAME_VARIABLES:
        throw new IllegalArgumentException(formula.toString());
      case DEFINE:
        stream.print("introduced(definition)");
        break;
      case CONJECTURE:
      case AXIOM:
        stream.printf(
            "file(%s,%s)",
            Etc.quote('\'', Path.of(formula.file).getFileName().toString()), formula.name);
        break;
      case NEGATE:
        stream.printf("inference(negate,[status(ceq)],[%s])", formula.from[0].name);
        break;
      default:
        stream.printf(
            "inference(%s,[status(", formula.inference.toString().toLowerCase(Locale.ROOT));

        // If a formula introduces new symbols, then it is only equisatisfiable
        // This happens during subformula renaming in CNF conversion
        var fromFuncs = new HashSet<>();
        for (var from : formula.from) Etc.collect(from.term(), a -> a instanceof Func, fromFuncs);
        var funcs = Etc.collect(formula.term(), a -> a instanceof Func);
        stream.print(fromFuncs.containsAll(funcs) ? "thm" : "esa");

        stream.print(")],[");
        for (var i = 0; i < formula.from.length; i++) {
          if (i > 0) stream.print(',');
          stream.print(formula.from[i].name);
        }
        stream.print("])");
        break;
    }
    stream.println(").");
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
    stream.print('(');
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) stream.print(',');
      print(a.get(i));
    }
    stream.print(')');
  }

  @SuppressWarnings("unchecked")
  private void print(Object a, List<Object> parent) {
    if (a instanceof List) {
      var a1 = (List) a;
      var op = a1.get(0);
      if (op instanceof Symbol) {
        if (needParens(a1, parent)) stream.print('(');
        switch ((Symbol) op) {
          case ALL:
            stream.print('!');
            quant(a1);
            break;
          case EXISTS:
            stream.print('?');
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
            stream.print('~');
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
        if (needParens(a1, parent)) stream.print(')');
        return;
      }
      print(op);
      args(a1);
      return;
    }
    if (a instanceof Func) {
      var name = a.toString();
      if (isWeird(name)) {
        stream.print(Etc.quote('\'', name));
        return;
      }
    }
    if (a instanceof Symbol) {
      print((Symbol) a);
      return;
    }
    if (a instanceof Boolean) stream.print('$');
    if (a instanceof String) {
      stream.print(Etc.quote('"', (String) a));
      return;
    }
    stream.print(a);
  }
}
