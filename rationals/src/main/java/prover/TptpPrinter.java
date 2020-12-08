package prover;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public final class TptpPrinter {

  // Temporary files
  private static final String IN_PATH = "/tmp/in.p";
  private static final String OUT_PATH = "/tmp/out.p";
  private static final String ERR_PATH = "/tmp/err.p";

  // Default: print to stdout
  public static TptpPrinter out = new TptpPrinter(new PrintWriter(System.out, true));

  // Can also print to file
  private final PrintWriter writer;

  public TptpPrinter(PrintWriter writer) {
    this.writer = writer;
  }

  private void args(Term term) {
    writer.print('(');
    for (var i = 1; i < term.size(); i++) {
      if (i > 1) {
        writer.print(',');
      }
      print(term.get(i), null);
    }
    writer.print(')');
  }

  public static String called(Term term) {
    switch (term.called()) {
      case ADD:
        return "$sum";
      case ALL:
        return "!";
      case AND:
        return "&";
      case CALL:
        return term.toString();
      case CEIL:
        return "$ceiling";
      case DIVIDE:
        return "$quotient";
      case DIVIDE_EUCLIDEAN:
        return "$quotient_e";
      case DIVIDE_FLOOR:
        return "$quotient_f";
      case DIVIDE_TRUNCATE:
        return "$quotient_t";
      case EQ:
        return "=";
      case EQV:
        return "<=>";
      case EXISTS:
        return "?";
      case FLOOR:
        return "$floor";
      case IS_INTEGER:
        return "$is_int";
      case IS_RATIONAL:
        return "$is_rat";
      case LESS:
        return "$lesseq";
      case LESS_EQ:
        return "$less";
      case MULTIPLY:
        return "$product";
      case NEGATE:
        return "$negate";
      case NOT:
        return "~";
      case OR:
        return "|";
      case REMAINDER_EUCLIDEAN:
        return "$remainder_e";
      case REMAINDER_FLOOR:
        return "$remainder_f";
      case REMAINDER_TRUNCATE:
        return "$remainder_t";
      case ROUND:
        return "$round";
      case SUBTRACT:
        return "$difference";
      case TO_INTEGER:
        return "$to_int";
      case TO_RATIONAL:
        return "$to_rat";
      case TO_REAL:
        return "$to_real";
      case TRUNCATE:
        return "$truncate";
    }
    throw new IllegalArgumentException(term.toString());
  }

  public static void debug(Formula formula) {
    var stack = Thread.currentThread().getStackTrace();
    out.writer.print(stack[2] + ": ");
    out.print(formula);
  }

  public static void debug(Term a) {
    var stack = Thread.currentThread().getStackTrace();
    out.writer.print(stack[2] + ": ");
    out.print(a, null);
    out.writer.println();
  }

  private void infix(Term term, String op) {
    for (var i = 1; i < term.size(); i++) {
      if (i > 1) {
        writer.print(op);
      }
      print(term.get(i), term);
    }
  }

  private static String nameVar(int i) {
    if (i < 26) {
      return Character.toString('A' + i);
    }
    return "Z" + (i - 25);
  }

  public static void nameVars(Term term) {
    var variables = term.variables();
    for (var x : variables) {
      x.setName(null);
    }
    var i = 0;
    for (var x : variables) {
      if (x.name() != null) {
        continue;
      }
      x.setName(nameVar(i++));
    }
  }

  public static boolean needParens(Term a, Term parent) {
    if (parent == null) {
      return false;
    }
    switch (a.op()) {
      case AND:
      case EQV:
      case OR:
        switch (parent.op()) {
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
    return false;
  }

  public void print(Formula formula) {
    print(formula, role(formula), source(formula));
  }

  public void print(List<? extends Formula> formulas) {
    for (var formula : formulas) {
      print(formula);
    }
  }

  private void print(Type type) {
    switch (type.kind()) {
      case BOOLEAN:
        writer.print("$o");
        return;
      case INDIVIDUAL:
        writer.print("$i");
        return;
      case INTEGER:
        writer.print("$int");
        return;
      case RATIONAL:
        writer.print("$rat");
        return;
      case REAL:
        writer.print("$real");
        return;
      default:
        throw new IllegalArgumentException(type.toString());
    }
  }

  private void print(Term term, Term parent) {
    switch (term.tag()) {
      case CONST_FALSE:
        writer.print("$false");
        return;
      case CONST_TRUE:
        writer.print("$true");
        return;
      case FUNC:
        {
          var name = term.toString();
          if (!Character.isLowerCase(name.charAt(0)) || weird(name)) {
            name = Util.quote('\'', name);
          }
          writer.print(name);
          return;
        }
      case LIST:
        break;
      default:
        writer.print(term.toString());
        return;
    }
    if (needParens(term, parent)) {
      writer.print('(');
    }
    switch (term.op()) {
      case ALL:
        writer.print('!');
        quant(term);
        break;
      case AND:
        infix(term, " & ");
        break;
      case EQ:
        infix(term, "=");
        break;
      case EQV:
        infix(term, " <=> ");
        break;
      case EXISTS:
        writer.print('?');
        quant(term);
        break;
      case NOT:
        if (term.get(0).op() == Op.EQ) {
          infix(term.get(1), "!=");
          break;
        }
        writer.print('~');
        print(term.get(1), term);
        break;
      case OR:
        infix(term, " | ");
        break;
      default:
        writer.print(called(term.get(0)));
        args(term);
        break;
    }
    if (needParens(term, parent)) {
      writer.print(')');
    }
  }

  private void print(Formula formula, String role, String source) {
    var sublanguage = sublanguage(formula);
    var term = formula.term();
    if ("cnf".equals(sublanguage)) {
      term = term.unquantify();
    }
    nameVars(term);
    writer.printf("%s(%s, %s, ", sublanguage, formula, role);
    print(term, null);
    if (source != null) {
      writer.print(", " + source);
    }
    writer.println(").");
  }

  private void quant(Term term) {
    writer.print('[');
    var params = term.get(1);
    for (var i = 0; i < params.size(); i++) {
      var x = params.get(i);
      if (i > 0) {
        writer.print(',');
      }
      writer.print(x);
      if (x.type() != Type.INDIVIDUAL) {
        writer.print(':');
        print(x.type());
      }
    }
    writer.print("]:");
    print(term.get(2), term);
  }

  public static String role(Formula formula) {
    switch (formula.szs()) {
      case CounterEquivalent:
        return "negated_conjecture";
      case LogicalData:
        if (formula instanceof FormulaTermInputConjecture) {
          return "conjecture";
        }
        if (formula.file() == null) {
          return "definition";
        }
        return "axiom";
    }
    return "plain";
  }

  private static SZS secondOpinion(String[] secondProver) throws IOException, InterruptedException {

    // Command
    var args = Arrays.copyOf(secondProver, secondProver.length + 1);
    args[secondProver.length] = IN_PATH;

    // Setup
    var builder = new ProcessBuilder(args);
    builder.redirectOutput(new File(OUT_PATH));
    builder.redirectError(new File(ERR_PATH));

    // Run
    var process = builder.start();
    process.waitFor();
    if ((process.exitValue() != 0) && (process.exitValue() != 1)) {
      Util.printFile(ERR_PATH);
      throw new IOException(String.join(" ", secondProver) + " returned " + process.exitValue());
    }

    // Result
    var pattern = Pattern.compile(".*SZS status (\\w+).*");
    for (var s : Files.readAllLines(Path.of(OUT_PATH), StandardCharsets.UTF_8)) {
      var matcher = pattern.matcher(s);
      if (matcher.matches()) {
        return SZS.valueOf(matcher.group(1));
      }
    }
    return null;
  }

  public static SZS secondOpinion(String[] secondProver, List<Clause> clauses)
      throws IOException, InterruptedException {

    // Printer
    var writer = new PrintWriter(IN_PATH, StandardCharsets.UTF_8);
    var printer = new TptpPrinter(writer);

    // Clauses
    for (var c : clauses) {
      printer.print(c, "axiom", null);
    }

    // Actually write the file
    writer.close();

    // Second opinion
    return secondOpinion(secondProver);
  }

  private static String source(Formula formula) {
    if (formula.szs() == SZS.LogicalData) {
      if (formula.file() == null) {
        return "introduced(definition)";
      }
      return String.format("file(%s,%s)", Util.quote('\'', formula.file()), formula);
    }
    return String.format(
        "inference(%s,[status(%s)],[%s])",
        formula.inference(),
        formula.szs().abbreviation().toLowerCase(Locale.ROOT),
        Arrays.stream(formula.from()).map(String::valueOf).collect(Collectors.joining(",")));
  }

  public static String sublanguage(Formula formula) {
    var ref =
        new Object() {
          boolean typed;
        };
    formula
        .term()
        .walk(
            (term, depth) -> {
              if ((term.type() != Type.BOOLEAN) && (term.type() != Type.INDIVIDUAL)) {
                ref.typed = true;
              }
            });
    if (ref.typed) {
      return "tff";
    }
    if (formula instanceof Clause) {
      return "cnf";
    }
    return "fof";
  }

  public static void verify(String[] secondProver, Clause conclusion)
      throws IOException, InterruptedException {
    var verified = 0;
    for (var formula : conclusion.proof()) {
      if (formula.szs() != SZS.Theorem) {
        continue;
      }

      // Printer
      var writer = new PrintWriter(IN_PATH, StandardCharsets.UTF_8);
      var printer = new TptpPrinter(writer);

      // Axioms
      for (var axiom : formula.from()) {
        printer.print(axiom, "axiom", null);
      }

      // Imply the derived formula?
      printer.print(formula, "conjecture", null);

      // Actually write the file
      writer.close();

      // Second opinion
      var szs = secondOpinion(secondProver);
      if (szs == null) {
        throw new IOException(
            String.format("Verify failed: %s gave no answer\n", String.join(" ", secondProver)));
      }
      switch (szs) {
        case ContradictoryAxioms:
        case Theorem:
        case Unsatisfiable:
          verified++;
          continue;
      }
      throw new IOException(
          String.format("Verify failed: %s said %s\n", String.join(" ", secondProver), szs));
    }
    if (Main.verbose) {
      System.out.printf("%% %d steps verified\n", verified);
    }
    new File(IN_PATH).delete();
    new File(OUT_PATH).delete();
    new File(ERR_PATH).delete();
  }

  public static boolean weird(String s) {
    for (var i = 0; i < s.length(); i++) {
      var c = s.charAt(i);
      if (Character.isLetterOrDigit(c)) {
        continue;
      }
      if (c == '_') {
        continue;
      }
      return true;
    }
    return false;
  }
}
