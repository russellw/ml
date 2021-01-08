package prover;

import io.vavr.collection.Seq;
import java.util.HashMap;

public final class TptpPrinter {
  private static final HashMap<Variable, String> variableNames = new HashMap<>();

  private static void print(Variable a) {
    var name = variableNames.get(a);
    if (name == null) {
      var i = variableNames.size();
      name = i < 26 ? Character.toString('A' + i) : "Z" + (i - 25);
      variableNames.put(a, name);
    }
    System.out.print(name);
  }

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
      case ALL:
        System.out.print("!");
        return;
      case AND:
        System.out.print("&");
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
      case EQUALS:
        System.out.print("=");
        return;
      case EQV:
        System.out.print("<=>");
        return;
      case EXISTS:
        System.out.print("?");
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
        System.out.print("rat");
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
      case NOT:
        System.out.print("~");
        return;
      case OR:
        System.out.print("|");
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

  private static boolean needParens(Seq a, Seq parent) {
    if (parent == null) return false;
    var op = a.head();
    if (op instanceof Symbol)
      switch ((Symbol) op) {
        case AND:
        case EQV:
        case OR:
          {
            var parentOp = parent.head();
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

  private static void infix(Seq a, String op) {
    for (var i = 1; i < a.size(); i++) {
      if (i > 1) System.out.print(op);
      print(a.get(i), a);
    }
  }

  private static void quant(Seq a) {
    System.out.print('[');
    var binding = (Seq) a.get(1);
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

  private static boolean isWeird(String s) {
    if (Character.isDigit(s.charAt(0))) return true;
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if (!(Character.isLetterOrDigit(c) || c == '_')) return true;
    }
    return false;
  }

  private static void print(Object a, Seq parent) {
    if (a instanceof Seq) {
      var a1 = (Seq) a;
      var op = a1.head();
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
            infix(a1, " & ");
            break;
          case OR:
            infix(a1, " | ");
            break;
          case NOT:
            if (Etc.head(a1.get(1)) == Symbol.EQUALS) {
              infix((Seq) a1.get(1), "!=");
              break;
            }
            System.out.print('~');
            print(a1.get(1), a1);
            break;
          case EQUALS:
            infix(a1, "=");
            break;
          case EQV:
            infix(a1, " <=> ");
            break;
          default:
            throw new IllegalArgumentException(a.toString());
        }
        if (needParens(a1, parent)) System.out.print(')');
        return;
      }
    }
    if (a instanceof Func) {
      var name = a.toString();
      if (isWeird(name)) {
        System.out.print(Etc.quote('\'', name));
        return;
      }
    }
    if (a instanceof Variable) {
      print((Variable) a);
      return;
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
